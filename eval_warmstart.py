"""eval_warmstart.py -- feasibility and rescue rate of every warm start (Table 1).

On held-out corridor instances, every arm is refined by the same SCvx solver with
the same iteration budget and judged by the same nonlinear certificate
(preflight_core.verify):
  * Cold          : SCvx from the SLERP initialization (with perturbed restarts
                    and a trust-weight sweep)
  * FM-Riem       : Riemannian flow matching on SO(3) (ours)
  * FM-Eucl       : Euclidean flow matching (quaternion as R^4)
  * DDPM-Eucl     : Euclidean diffusion
  * Regress       : deterministic regression
  * NN-retrieval  : nearest training trajectory by conditioning distance

Rescue = fraction of the instances Cold fails that the arm solves.

Table 1 (n=500, seeds 2025/2026/2027, 2 deg gate):

    python3 eval_warmstart.py --seeds 2025 2026 2027 --n 500 \
        --fm_riem corridor_fmriem.pt --fm_eucl corridor_fmeucl.pt \
        --ddpm corridor_ddpm.pt --regress corridor_regress.pt --nn corridor.pkl --device cuda

Add --tol_att 5 for the relaxed-gate robustness study.
"""
import argparse
import numpy as np
import torch
import preflight_core
from preflight_core import verify
from fast_scvx import FastSCvx
from generate_corridor import make_corridor, cold_best


def eval_three_way(test_insts, S, warm_init_fns, n_iter=18, trust_weight=8.0, quiet=False):
    arms = ['cold'] + list(warm_init_fns)
    agg = {a: {'feas': 0, 'costs': []} for a in arms}
    cold_fail_idx = []
    rows = []
    for i, inst in enumerate(test_insts):
        row = {}
        cb = cold_best(S, inst, seed=1000 + i)
        if cb is not None:
            agg['cold']['feas'] += 1; agg['cold']['costs'].append(cb[0]); row['cold'] = cb[0]
        else:
            row['cold'] = None; cold_fail_idx.append(i)
        for name, fn in warm_init_fns.items():
            xi_init = fn(inst)
            if xi_init is None:
                row[name] = None; continue
            r = S.solve(inst, xi_init=xi_init, n_iter=n_iter, trust_weight=trust_weight)
            v = verify(inst, r['tau'])
            if v['feasible']:
                agg[name]['feas'] += 1; agg[name]['costs'].append(r['costs'][-1]); row[name] = r['costs'][-1]
            else:
                row[name] = None
        rows.append(row)

    n = len(test_insts)
    for name in warm_init_fns:                       # record rescue count for aggregation
        agg[name]['rescue'] = sum(1 for i in cold_fail_idx if rows[i][name] is not None)
    agg['_n_cold_fail'] = len(cold_fail_idx)
    if quiet:
        return agg, rows
    print(f"\n{'arm':>10} {'feasible':>10} {'median_cost':>12}  {'feasible_where_cold_failed':>26}")
    for a in arms:
        feas = agg[a]['feas']; costs = agg[a]['costs']
        med = f"{np.median(costs):.3f}" if costs else "-"
        if a == 'cold':
            rescue = '-'
        else:
            rescue = f"{agg[a]['rescue']}/{len(cold_fail_idx)}"
        print(f"{a:>10} {feas}/{n:>8} {med:>12}  {str(rescue):>26}")
    print("\nrescue = 'feasible_where_cold_failed': the warm arm solving problems cold cannot.")
    return agg, rows


# ----- FM warm-start machinery -----
_PAD_DIR = np.array([0., 0., 1.])     # inactive-cone dir (cos_exc=1.0 -> no-op)

def make_cond(inst, m_max=None):
    import torch
    cones = list(inst['cones'])
    if m_max is not None:
        if len(cones) < m_max:
            cones = cones + [(_PAD_DIR.copy(), 1.0)] * (m_max - len(cones))
        elif len(cones) > m_max:
            cones = cones[:m_max]
    cone_dirs = np.concatenate([c[0] for c in cones])
    cos_excs = np.array([c[1] for c in cones])
    cond = np.concatenate([inst['q0'], inst['q_tgt'], cone_dirs, cos_excs]).astype(np.float32)
    return torch.from_numpy(cond).unsqueeze(0)

def load_fm(path, kind, device='cpu', reanchor=False, no_cvf=False):
    """kind in {'riem','eucl'}. Returns a fn(inst)->xi_init (3,T).
    reanchor=True re-bases the sampled path to start at its own first state
    (so xi[:,0]=0), the fair fix for a warm-start whose start drifts from q0.
    No-op for the Riemannian sampler, which already anchors the start."""
    import torch
    from backbone_v2 import JointVelocityNetV2
    from preflight_core import log_so3, qmul, qconj, qnorm
    ck = torch.load(path, map_location=device)
    m_max = (ck['cond_dim'] - 8) // 4
    net = JointVelocityNetV2(seq_len=ck['seq_len'], cond_dim=ck['cond_dim'], hidden=ck['hidden'],
                             n_layers=ck['n_layers'], n_heads=ck['n_heads'])
    T = ck['seq_len']
    if kind == 'riem':
        from joint_fm import sample_joint
        net.load_state_dict(ck['state_dict']); net.to(device).eval()

        def sampler(cond):
            return sample_joint(net, cond, ck['stats'], seq_len=T, n_steps=50, device=device,
                                cvf_initial_attitude=not no_cvf, cvf_final_attitude=not no_cvf,
                                cvf_zero_rates=not no_cvf)
    elif kind == 'ddpm':
        from ddpm_eucl import DDPMHead, sample_ddpm
        model = DDPMHead(net); model.load_state_dict(ck['state_dict']); model.to(device).eval()
        def sampler(cond):
            return sample_ddpm(model, cond, ck['stats'], seq_len=T, device=device)
    elif kind == 'regress':
        from regress_baseline import RegressionHead, sample_regress
        model = RegressionHead(net); model.load_state_dict(ck['state_dict']); model.to(device).eval()
        def sampler(cond):
            return sample_regress(model, cond, ck['stats'], seq_len=T, device=device)
    else:
        from eucl_fm import EuclHead, sample_eucl
        model = EuclHead(net); model.load_state_dict(ck['state_dict']); model.to(device).eval()
        def sampler(cond):
            return sample_eucl(model, cond, ck['stats'], seq_len=T, n_steps=50, device=device)

    def init_fn(inst):
        cond = make_cond(inst, m_max=m_max).to(device)
        samp = sampler(cond)[0].detach().cpu().numpy()        # (T, 10)
        q = samp[:, 0:4].copy()
        for k in range(1, len(q)):                            # sign-continuity unwrap
            if np.dot(q[k], q[k-1]) < 0: q[k] = -q[k]
        base = qnorm(q[0]) if reanchor else inst['q0']
        xi = np.stack([log_so3(qmul(qconj(base), qnorm(q[k]))) for k in range(T)], axis=1)
        return xi
    return init_fn


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--M', type=int, default=5)
    p.add_argument('--tau', type=float, default=0.15)
    p.add_argument('--N', type=int, default=60)
    p.add_argument('--sep', type=float, default=120.0)
    p.add_argument('--half', type=float, default=20.0)
    p.add_argument('--n', type=int, default=500)
    p.add_argument('--seed', type=int, default=2025, help='held-out seed (distinct from the training seed 11)')
    p.add_argument('--seeds', type=int, nargs='+', default=None,
                   help='multiple held-out seeds -> reports mean+/-std feasibility and rescue across seeds')
    p.add_argument('--fm_riem', default=None, help='FM-Riem checkpoint (corridor_fmriem.pt)')
    p.add_argument('--fm_eucl', default=None, help='FM-Eucl checkpoint (corridor_fmeucl.pt)')
    p.add_argument('--ddpm', default=None, help='DDPM-Eucl checkpoint (corridor_ddpm.pt)')
    p.add_argument('--regress', default=None, help='deterministic regression checkpoint (*_regress.pt)')
    p.add_argument('--nn', default=None, help='training pkl for nearest-neighbor retrieval baseline')
    p.add_argument('--device', default='cpu')
    p.add_argument('--n_iter', type=int, default=18, help='SCvx iterations per warm arm (same for every arm)')
    p.add_argument('--trust_weight', type=float, default=8.0)
    p.add_argument('--tol_att', type=float, default=2.0, help='terminal-attitude gate of the certificate (deg)')
    args = p.parse_args()
    preflight_core.TOL_ATT_DEG = args.tol_att
    torch.manual_seed(args.seed)          # make FM sampling reproducible
    S = FastSCvx(N=args.N, ncone=args.M, dt=0.6, tau_max=args.tau, J=np.diag([10., 12., 15.]))

    def build_arms():
        arms = {}
        if args.fm_riem: arms['FM-Riem'] = load_fm(args.fm_riem, 'riem', args.device)
        if args.fm_eucl:
            arms['FM-Eucl'] = load_fm(args.fm_eucl, 'eucl', args.device, reanchor=True)
        if args.ddpm: arms['DDPM-Eucl'] = load_fm(args.ddpm, 'ddpm', args.device, reanchor=True)
        if args.regress: arms['Regress'] = load_fm(args.regress, 'regress', args.device, reanchor=True)
        if args.nn:
            from nn_baseline import make_nn_init
            arms['NN-retrieval'] = make_nn_init(args.nn, reanchor=True)
        if not arms:
            raise SystemExit('give at least one warm-start arm (--fm_riem/--fm_eucl/--ddpm/--regress/--nn)')
        return arms

    if args.seeds:                                   # multi-seed CI mode
        arms = build_arms(); arm_names = ['cold'] + list(arms)
        feas = {a: [] for a in arm_names}; resc = {a: [] for a in list(arms)}
        for sd in args.seeds:
            rng = np.random.default_rng(sd)
            test = [make_corridor(rng, args.M, args.tau, args.N, args.sep, args.half) for _ in range(args.n)]
            agg, _ = eval_three_way(test, S, arms, n_iter=args.n_iter, trust_weight=args.trust_weight, quiet=True)
            for a in arm_names: feas[a].append(agg[a]['feas'])
            for a in arms: resc[a].append((agg[a]['rescue'], agg['_n_cold_fail']))
        print(f"=== multi-seed eval: M={args.M} tau={args.tau} N={args.N}, n={args.n}, "
              f"seeds={args.seeds} ===")
        print(f"{'arm':>12} {'feasible (mean+/-std)':>24} {'rescue% (mean+/-std)':>22}")
        for a in arm_names:
            fmean, fstd = np.mean(feas[a]), np.std(feas[a])
            if a == 'cold':
                print(f"{a:>12} {f'{fmean:.1f}+/-{fstd:.1f} / {args.n}':>24} {'-':>22}")
            else:
                rates = [100.0 * r / d if d else 0.0 for r, d in resc[a]]
                print(f"{a:>12} {f'{fmean:.1f}+/-{fstd:.1f} / {args.n}':>24} "
                      f"{f'{np.mean(rates):.1f}+/-{np.std(rates):.1f}%':>22}")
        return

    rng = np.random.default_rng(args.seed)
    test = [make_corridor(rng, args.M, args.tau, args.N, args.sep, args.half) for _ in range(args.n)]
    arms = build_arms()
    print(f"=== warm-start eval: M={args.M} tau={args.tau} N={args.N}, {args.n} held-out, arms={list(arms)} ===")
    eval_three_way(test, S, arms, n_iter=args.n_iter, trust_weight=args.trust_weight)


if __name__ == '__main__':
    main()
