"""efficiency_compare.py -- iterations, runtime and control cost on the common set (Table 2).

For each held-out instance and each arm, find the minimum SCvx iteration count at
which the nonlinear certificate passes (iterations to first certified solution),
time that solve, and record the converged control cost sum ||tau_k||^2 dt. Medians
are reported on the COMMON-SOLVABLE set (instances every compared arm solves), so
the comparison is not confounded by an arm only solving the easy instances. The
paired per-instance cost ratio Cold/arm is reported as well (>1 = arm cheaper).

    python3 efficiency_compare.py --M 5 --tau 0.15 --N 60 --n 120 \
        --fm_riem corridor_fmriem.pt --regress corridor_regress.pt --nn corridor.pkl --device cuda
"""
import argparse, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
import preflight_core
from preflight_core import verify
from fast_scvx import FastSCvx
from generate_corridor import make_corridor
from eval_warmstart import load_fm

K_GRID = [1, 2, 3, 4, 6, 8, 12, 18]      # iteration budgets probed, ascending


def iters_to_feasible(S, inst, xi0):
    """Smallest k in K_GRID with a verifier-feasible solve. Returns
    (k, cost, wall) or (None, None, None). xi0 sampled once, reused per k."""
    for k in K_GRID:
        t0 = time.time()
        r = S.solve(inst, xi_init=xi0, n_iter=k, trust_weight=8.0)
        wall = time.time() - t0
        if r['tau'] is not None and verify(inst, r['tau'])['feasible']:
            return k, r['costs'][-1], wall
    return None, None, None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--M', type=int, default=5); p.add_argument('--tau', type=float, default=0.15)
    p.add_argument('--N', type=int, default=60); p.add_argument('--n', type=int, default=120)
    p.add_argument('--sep', type=float, default=120.0); p.add_argument('--half', type=float, default=20.0)
    p.add_argument('--seed', type=int, default=2025)
    p.add_argument('--fm_riem', default=None); p.add_argument('--fm_eucl', default=None)
    p.add_argument('--ddpm', default=None)
    p.add_argument('--regress', default=None); p.add_argument('--nn', default=None)
    p.add_argument('--device', default='cpu')
    p.add_argument('--tol_att', type=float, default=2.0, help='terminal-attitude gate of the certificate (deg)')
    a = p.parse_args()
    preflight_core.TOL_ATT_DEG = a.tol_att

    S = FastSCvx(N=a.N, ncone=a.M, dt=0.6, tau_max=a.tau, J=np.diag([10., 12., 15.]))
    rng = np.random.default_rng(a.seed)
    test = [make_corridor(rng, a.M, a.tau, a.N, a.sep, a.half) for _ in range(a.n)]

    arms = {'cold': None}
    if a.fm_riem: arms['FM-Riem'] = load_fm(a.fm_riem, 'riem', a.device)
    if a.fm_eucl: arms['FM-Eucl'] = load_fm(a.fm_eucl, 'eucl', a.device, reanchor=True)
    if a.ddpm: arms['DDPM-Eucl'] = load_fm(a.ddpm, 'ddpm', a.device, reanchor=True)
    if a.regress: arms['Regress'] = load_fm(a.regress, 'regress', a.device, reanchor=True)
    if a.nn:
        from nn_baseline import make_nn_init
        arms['NN-retrieval'] = make_nn_init(a.nn, reanchor=True)

    # per-instance results
    rec = {name: [] for name in arms}        # list of (k, cost, wall) or None
    for inst in test:
        for name, fn in arms.items():
            xi0 = fn(inst) if fn is not None else None
            rec[name].append(iters_to_feasible(S, inst, xi0))

    solved = {name: np.array([r[0] is not None for r in rec[name]]) for name in arms}
    # common set = intersection of arms that actually solve a meaningful fraction
    # (an arm solving ~nothing, e.g. FM-Eucl, must not zero out the common set)
    meaningful = [name for name in arms if solved[name].mean() > 0.05]
    excluded = [name for name in arms if name not in meaningful]
    common = np.ones(a.n, dtype=bool)
    for name in meaningful:
        common &= solved[name]
    nc = int(common.sum())
    print(f"=== efficiency on common-solvable set: M={a.M} tau={a.tau} N={a.N}, "
          f"{nc}/{a.n} solved by ALL of {meaningful} ===")
    print(f"\n{'arm':>10} {'feas_on_common':>14} {'med_iters':>10} {'med_wall_s':>11} {'med_cost':>10}")
    idx = np.where(common)[0]
    for name in meaningful:
        iters = [rec[name][i][0] for i in idx]
        wall = [rec[name][i][2] for i in idx]
        cost = [rec[name][i][1] for i in idx]
        print(f"{name:>10} {nc:>14} {np.median(iters):>10.1f} "
              f"{np.median(wall):>11.3f} {np.median(cost):>10.4f}")
    print("\nmed_iters  = SCvx iterations to first verifier-feasible solve (lower = better init)")
    print("med_cost   = converged objective on the SAME instances (apples-to-apples)")

    # paired per-instance cost ratio vs cold (instance by instance)
    if 'cold' in meaningful:
        cold_cost = np.array([rec['cold'][i][1] for i in idx])
        print(f"\npaired cost ratio cold/arm across the {nc} common instances "
              f"(>1 means arm is cheaper):")
        for name in meaningful:
            if name == 'cold':
                continue
            arm_cost = np.array([rec[name][i][1] for i in idx])
            ratio = cold_cost / arm_cost
            wins = int((arm_cost < cold_cost - 1e-9).sum())
            print(f"  {name:>10}: median ratio {np.median(ratio):.3f}  "
                  f"(cheaper on {wins}/{nc}, mean ratio {ratio.mean():.3f})")

    print(f"\noverall solved (each arm, all {a.n}):")
    for name in arms:
        tag = "  (excluded from common: solves <5%)" if name in excluded else ""
        print(f"  {name:>10}: {int(solved[name].sum())}/{a.n}{tag}")


if __name__ == '__main__':
    main()
