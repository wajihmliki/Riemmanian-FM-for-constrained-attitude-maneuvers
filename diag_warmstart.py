"""diag_warmstart.py -- why the Euclidean generative baselines fail (Fig. 4 discussion).

For each generative warm start (FM-Riem, FM-Eucl, DDPM-Eucl), on held-out corridor
instances:
  (1) smoothness of the GENERATED attitude path: mean/max inter-node geodesic
      rotation of the extracted xi (a SLERP path is a few deg/node).
  (2) feasibility vs SCvx iteration budget (n_iter in {18, 30, 50}). If an arm
      stays at zero as the budget grows, the failure comes from the
      initialization, not from insufficient optimization.

    python3 diag_warmstart.py --M 5 --tau 0.15 --N 60 --n 60 \
        --fm_riem corridor_fmriem.pt --fm_eucl corridor_fmeucl.pt \
        --ddpm corridor_ddpm.pt --device cuda
"""
import argparse, warnings
import numpy as np
warnings.filterwarnings("ignore")
from preflight_core import verify
from fast_scvx import FastSCvx
from generate_corridor import make_corridor
from eval_warmstart import load_fm


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--M', type=int, default=5); p.add_argument('--tau', type=float, default=0.15)
    p.add_argument('--N', type=int, default=60); p.add_argument('--n', type=int, default=60)
    p.add_argument('--sep', type=float, default=120.0); p.add_argument('--half', type=float, default=20.0)
    p.add_argument('--seed', type=int, default=2025)
    p.add_argument('--fm_riem', default=None); p.add_argument('--fm_eucl', default=None)
    p.add_argument('--ddpm', default=None); p.add_argument('--device', default='cpu')
    p.add_argument('--iters', type=int, nargs='+', default=[18, 30, 50])
    p.add_argument('--trust_weight', type=float, default=8.0)
    a = p.parse_args()

    S = FastSCvx(N=a.N, ncone=a.M, dt=0.6, tau_max=a.tau, J=np.diag([10., 12., 15.]))
    rng = np.random.default_rng(a.seed)
    insts = [make_corridor(rng, a.M, a.tau, a.N, a.sep, a.half) for _ in range(a.n)]

    arms = {}
    if a.fm_riem: arms['FM-Riem'] = load_fm(a.fm_riem, 'riem', a.device)
    if a.fm_eucl: arms['FM-Eucl'] = load_fm(a.fm_eucl, 'eucl', a.device, reanchor=True)
    if a.ddpm:    arms['DDPM-Eucl'] = load_fm(a.ddpm, 'ddpm', a.device, reanchor=True)

    print(f"diagnostic on {a.n} instances (M={a.M}, tau={a.tau}, N={a.N})\n")
    print("(1) GENERATED-PATH SMOOTHNESS (inter-step geodesic angle of the warm-start xi)")
    print(f"{'arm':>11} {'mean_deg':>9} {'median_deg':>11} {'max_deg':>9}   reading")
    xis = {name: [] for name in arms}
    for name, fn in arms.items():
        means, maxes = [], []
        for inst in insts:
            xi = fn(inst); xis[name].append(xi)
            step = np.degrees(np.linalg.norm(np.diff(xi, axis=1), axis=0))
            means.append(step.mean()); maxes.append(step.max())
        mm = float(np.mean(means)); md = float(np.median(means)); mx = float(np.mean(maxes))
        tag = "smooth" if mm < 10 else ("jagged" if mm < 60 else "near-random-walk")
        print(f"{name:>11} {mm:>9.1f} {md:>11.1f} {mx:>9.1f}   {tag}")
    print("  (a smooth slerp is ~2-4 deg/step; >60 deg/step is essentially noise on SO(3))\n")

    print("(2) FEASIBILITY vs SCvx ITERATION BUDGET (is the budget the handicap?)")
    header = "       arm " + "".join(f"  n_iter={k:<3}" for k in a.iters)
    print(header)
    for name, fn in arms.items():
        cells = []
        for k in a.iters:
            feas = 0
            for j, inst in enumerate(insts):
                r = S.solve(inst, xi_init=xis[name][j], n_iter=k, trust_weight=a.trust_weight)
                if r['tau'] is not None and verify(inst, r['tau'])['feasible']:
                    feas += 1
            cells.append(f"{feas:>3}/{a.n:<3}")
        print(f"{name:>10} " + "".join(f"  {c:>9}" for c in cells))
    print("\nan arm that stays ~0 across all budgets fails because of its initialization, not the budget.")


if __name__ == '__main__':
    main()
