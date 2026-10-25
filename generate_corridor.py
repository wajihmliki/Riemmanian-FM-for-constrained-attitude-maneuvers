"""generate_corridor.py -- dense multi-cone "corridor" attitude problems and
certified demonstrations.

M keep-out cones are placed as an alternating slalom along a ~120 deg slew so the
direct SLERP route is blocked and a nontrivial detour is required. A random global
orientation gives training diversity.

Each instance is solved by SCvx from several structured initializations; the
lowest-cost trajectory that passes the nonlinear certificate (preflight_core.verify)
is kept, and unsolved instances are discarded, so every saved demonstration is
dynamically valid.

Modes:
  --preflight : check that cold fails / structured inits succeed on a few problems
  (default)   : generate certified demonstrations

    python3 generate_corridor.py --M 5 --tau 0.15 --N 60 --n_unique 2000 --out corridor.pkl
"""
import argparse, pickle
import numpy as np
from preflight_core import (verify, log_so3, qmul, qconj, qnorm, exp_so3, rotmat,
                            scvx_solve)
from fast_scvx import FastSCvx


def rand_rot(rng):
    v = rng.normal(size=3); v /= np.linalg.norm(v); a = rng.uniform(0, 2*np.pi)
    return rotmat(exp_so3(a*v))

def quat_z_to(d):
    z = np.array([0., 0, 1.]); d = d/np.linalg.norm(d); c = np.clip(z@d, -1, 1)
    ax = np.cross(z, d); s = np.linalg.norm(ax)
    if s < 1e-9:
        return np.array([1., 0, 0, 0]) if c > 0 else np.array([0., 1, 0, 0])
    return exp_so3(np.arccos(c) * ax/s)

def make_corridor(rng, M=5, tau=0.15, N=60, sep_deg=120.0, half_deg=20.0):
    R = rand_rot(rng)
    bs0 = R @ np.array([np.cos(np.deg2rad(-sep_deg/2)), 0, np.sin(np.deg2rad(-sep_deg/2))])
    bsT = R @ np.array([np.cos(np.deg2rad( sep_deg/2)), 0, np.sin(np.deg2rad( sep_deg/2))])
    bs0 /= np.linalg.norm(bs0); bsT /= np.linalg.norm(bsT)
    lateral = np.cross(bs0, bsT); lateral /= np.linalg.norm(lateral)
    perp = np.cross(np.cross(bs0, bsT), bs0); perp /= np.linalg.norm(perp)
    cones = []
    for m in range(M):
        frac = (m+1)/(M+1)
        mid = bs0*np.cos(frac*np.deg2rad(sep_deg)) + perp*np.sin(frac*np.deg2rad(sep_deg))
        off = (-1)**m * np.deg2rad(rng.uniform(7, 14))
        cdir = mid*np.cos(off) + lateral*np.sin(off); cdir /= np.linalg.norm(cdir)
        cones.append((cdir, np.cos(np.deg2rad(half_deg))))
    q0 = qnorm(qmul(quat_z_to(bs0), exp_so3(np.array([0, 0, rng.uniform(-0.3, 0.3)]))))
    q_tgt = qnorm(qmul(quat_z_to(bsT), exp_so3(np.array([0, 0, rng.uniform(-0.3, 0.3)]))))
    return dict(J=np.diag([10., 12., 15.]), dt=0.6, N=N, b_body=np.array([0., 0, 1.]),
                tau_max=tau, w0=np.zeros(3), xi_target=log_so3(qmul(qconj(q0), q_tgt)),
                q0=q0, q_tgt=q_tgt, cones=cones, _lat=lateral)

def basket(xit, N, amp=0.8):
    base = np.outer(xit, np.linspace(0, 1, N+1)); out = [base.copy()]
    b = np.sin(np.linspace(0, np.pi, N+1))
    for ax in range(3):
        for s in (+1, -1):
            d = base.copy(); d[ax] += s*amp*b; d[:, 0] = 0; d[:, -1] = xit; out.append(d)
    return out

def cold_best(S, inst, K=3, seed=0):
    rng = np.random.default_rng(seed); xit = inst['xi_target']; N = inst['N']
    base = np.outer(xit, np.linspace(0, 1, N+1)); best = None
    for k in range(K):
        init = base.copy() if k == 0 else base + rng.normal(scale=0.13, size=base.shape)
        init[:, 0] = 0; init[:, -1] = xit
        for tw in (5., 15.):
            r = S.solve(inst, xi_init=init, n_iter=18, trust_weight=tw)
            if verify(inst, r['tau'])['feasible'] and (best is None or r['costs'][-1] < best[0]):
                best = (r['costs'][-1], r['xi'], r['om'], r['tau'])
    return best

def struct_best(S, inst):
    best = None
    for init in basket(inst['xi_target'], inst['N']):
        r = S.solve(inst, xi_init=init, n_iter=18, trust_weight=8.0)
        if verify(inst, r['tau'])['feasible'] and (best is None or r['costs'][-1] < best[0]):
            best = (r['costs'][-1], r['xi'], r['om'], r['tau'])
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--preflight', action='store_true')
    p.add_argument('--M', type=int, default=5)
    p.add_argument('--tau', type=float, default=0.15)
    p.add_argument('--N', type=int, default=60)
    p.add_argument('--sep', type=float, default=120.0)
    p.add_argument('--half', type=float, default=20.0)
    p.add_argument('--n', type=int, default=5)
    p.add_argument('--n_unique', type=int, default=2000)
    p.add_argument('--seed', type=int, default=11)
    p.add_argument('--out', default='corridor.pkl')
    args = p.parse_args()
    S = FastSCvx(N=args.N, ncone=args.M, dt=0.6, tau_max=args.tau, J=np.diag([10., 12., 15.]))
    rng = np.random.default_rng(args.seed)

    if args.preflight:
        print(f"=== corridor preflight: M={args.M} cones, tau={args.tau}, N={args.N}, {args.n} problems ===")
        wins = 0; both = 0
        for i in range(args.n):
            inst = make_corridor(rng, args.M, args.tau, args.N, args.sep, args.half)
            cb = cold_best(S, inst, seed=100+i); sb = struct_best(S, inst)
            cf = cb is not None; sf = sb is not None
            tag = 'GOOD WINS' if (not cf and sf) else ('cold ok' if cf else 'both fail')
            if not cf and sf: wins += 1
            if not cf and not sf: both += 1
            cc = f"{cb[0]:.3f}" if cf else "INFEAS"; sc = f"{sb[0]:.3f}" if sf else "INFEAS"
            print(f"  prob {i}: cold={cc:>8}  struct={sc:>8}  -> {tag}")
        print(f"\ncold fails / structured succeeds: {wins}/{args.n}   both-fail (too tight): {both}/{args.n}")
        print(f"GO for corridor warm-start regime: {wins > args.n/2}")
        return

    # generation: produce physically-verified corridor demos
    from tqdm import tqdm
    trajs, conds = [], []
    pbar = tqdm(total=args.n_unique, desc='corridor')
    n_attempt = 0
    while len(conds) < args.n_unique:
        n_attempt += 1
        inst = make_corridor(rng, args.M, args.tau, args.N, args.sep, args.half)
        sol = struct_best(S, inst)            # demonstrator uses good init
        if sol is None:
            continue
        _, xi, om, tau = sol
        q0 = inst['q0']
        quats = np.stack([qnorm(qmul(q0, exp_so3(xi[:, k]))) for k in range(args.N+1)], axis=1)
        tau_pad = np.concatenate([tau, tau[:, -1:]], axis=1)
        cone_dirs = np.concatenate([c[0] for c in inst['cones']])
        cos_excs = np.array([c[1] for c in inst['cones']])
        cond = np.concatenate([inst['q0'], inst['q_tgt'], cone_dirs, cos_excs]).astype(np.float32)
        trajs.append(np.concatenate([quats, om, tau_pad], axis=0).astype(np.float32))
        conds.append(cond); pbar.update(1)
    pbar.close()
    data = dict(trajectories=np.stack(trajs), conditions=np.stack(conds),
                meta=dict(N=args.N, tf=args.N*0.6, tau_max=args.tau, M_cones=args.M,
                          J=np.diag([10., 12., 15.]).tolist(), b_body=[0., 0, 1.],
                          cond_layout='q0(4) q_target(4) cone_dirs(3M) cos_excs(M)',
                          generator='corridor_corrected_verified'))
    pickle.dump(data, open(args.out, 'wb'))
    print(f"\nSaved {len(trajs)} verified corridor demos -> {args.out}  (feasibility {args.n_unique/n_attempt:.1%})")


if __name__ == '__main__':
    main()
