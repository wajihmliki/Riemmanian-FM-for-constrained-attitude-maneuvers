"""horizon_general.py -- transfer to longer planning horizons (Table 5).

The FM is trained at N=60 (seq_len 61) and conditions only on (q0, target, cones),
none of which depend on N. For a longer-horizon problem the model is sampled at its
native length and the route is re-timed to the target grid, then used to warm-start
SCvx. No retraining.

    python3 horizon_general.py --N_list 90 120 --M 5 --tau 0.15 --n 80 \
        --fm_riem corridor_fmriem.pt --device cuda
"""
import argparse, warnings
import numpy as np
warnings.filterwarnings("ignore")
from preflight_core import verify
from fast_scvx import FastSCvx
from generate_corridor import make_corridor, cold_best
from eval_warmstart import load_fm


def interp_xi(xi_full, H):
    """Re-time a full-length warm-start (3,Tf) to a horizon of H intervals -> (3,H+1)."""
    Tf = xi_full.shape[1]
    src = np.linspace(0, 1, Tf); dst = np.linspace(0, 1, H + 1)
    return np.stack([np.interp(dst, src, xi_full[c]) for c in range(3)], axis=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--N_list', type=int, nargs='+', default=[90, 120])
    p.add_argument('--M', type=int, default=5); p.add_argument('--tau', type=float, default=0.15)
    p.add_argument('--n', type=int, default=80)
    p.add_argument('--sep', type=float, default=120.0); p.add_argument('--half', type=float, default=20.0)
    p.add_argument('--seed', type=int, default=2025)
    p.add_argument('--fm_riem', required=True); p.add_argument('--device', default='cpu')
    a = p.parse_args()

    fn = load_fm(a.fm_riem, 'riem', a.device)     # samples at the model's native length
    Jm = np.diag([10., 12., 15.])
    print(f"FM trained at N=60; re-timed to each target horizon (no retraining).\n")
    print(f"{'N':>4} {'maneuver_s':>11} {'cold_feas':>10} {'FM_feas':>9} {'rescue(FM|~cold)':>17}")
    for N in a.N_list:
        S = FastSCvx(N=N, ncone=a.M, dt=0.6, tau_max=a.tau, J=Jm)
        rng = np.random.default_rng(a.seed)
        insts = [make_corridor(rng, a.M, a.tau, N, a.sep, a.half) for _ in range(a.n)]
        cold_ok = np.zeros(a.n, bool); fm_ok = np.zeros(a.n, bool)
        for i, inst in enumerate(insts):
            cb = cold_best(S, inst, seed=1000 + i)
            cold_ok[i] = cb is not None
            xi = interp_xi(fn(inst), N)            # re-time native route to this horizon
            r = S.solve(inst, xi_init=xi, n_iter=18, trust_weight=8.0)
            fm_ok[i] = (r['tau'] is not None) and verify(inst, r['tau'])['feasible']
        rescue = int((fm_ok & ~cold_ok).sum()); ncf = int((~cold_ok).sum())
        print(f"{N:>4} {N*0.6:>11.1f} {int(cold_ok.sum()):>6}/{a.n:<3} "
              f"{int(fm_ok.sum()):>6}/{a.n:<3} {rescue}/{ncf:>14}")
    print("\nnote: fixed dt=0.6, so larger N = longer maneuver (more control time).")


if __name__ == '__main__':
    main()
