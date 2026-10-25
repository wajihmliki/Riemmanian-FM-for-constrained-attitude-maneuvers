"""difficulty_sweep.py -- Cold vs FM-Riem across torque limit and cone count (Tables 3-4).

At each (M, tau) cell, reports Cold and FM-Riem feasibility at the 2 deg gate and
the rescue count (FM-Riem solves among Cold's failures).

The model can be applied to any M up to the number of cone slots in its
conditioning (cond_dim = 8 + 4*M_max); unused slots are padded with inactive
cones (cos_exc = 1). Use the variable-cone model for the cone-count sweep.

Table 3 (torque, M=5, n=80):
    python3 difficulty_sweep.py --M_list 5 --tau_list 0.10 0.12 0.15 0.20 0.30 \
        --n 80 --fm_riem corridor_fmriem.pt --device cuda
Table 4 (cones, n=80):
    python3 difficulty_sweep.py --M_list 1 2 3 4 5 6 7 8 --tau_list 0.15 \
        --n 80 --fm_riem corridor_varM_fmriem.pt --device cuda
"""
import argparse, warnings
import numpy as np
warnings.filterwarnings("ignore")
import torch
from preflight_core import verify
from fast_scvx import FastSCvx
from generate_corridor import make_corridor, cold_best
from eval_warmstart import load_fm


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--M_list', type=int, nargs='+', default=[5])
    p.add_argument('--tau_list', type=float, nargs='+', default=[0.10, 0.12, 0.15, 0.20, 0.30])
    p.add_argument('--N', type=int, default=60); p.add_argument('--n', type=int, default=80)
    p.add_argument('--sep', type=float, default=120.0); p.add_argument('--half', type=float, default=20.0)
    p.add_argument('--seed', type=int, default=2025)
    p.add_argument('--fm_riem', default=None); p.add_argument('--device', default='cpu')
    a = p.parse_args()

    M_model = None; fn = None
    if a.fm_riem:
        ck = torch.load(a.fm_riem, map_location='cpu')
        M_model = (ck['cond_dim'] - 8) // 4
        fn = load_fm(a.fm_riem, 'riem', a.device)
        print(f"FM-Riem covers M<=M_max={M_model} (cond_dim={ck['cond_dim']}); "
              f"applied at all M<={M_model} cells.\n")

    Jm = np.diag([10., 12., 15.])
    print(f"{'M':>3} {'tau':>6} {'cold_feas':>10} {'FM_feas':>9} {'rescue(FM|~cold)':>17} {'note':>14}")
    for M in a.M_list:
        for tau in a.tau_list:
            S = FastSCvx(N=a.N, ncone=M, dt=0.6, tau_max=tau, J=Jm)
            rng = np.random.default_rng(a.seed)
            insts = [make_corridor(rng, M, tau, a.N, a.sep, a.half) for _ in range(a.n)]
            apply_fm = (fn is not None and M <= M_model)
            cold_ok = np.zeros(a.n, bool); fm_ok = np.zeros(a.n, bool)
            for i, inst in enumerate(insts):
                cb = cold_best(S, inst, seed=1000 + i)
                cold_ok[i] = cb is not None
                if apply_fm:
                    r = S.solve(inst, xi_init=fn(inst), n_iter=18, trust_weight=8.0)
                    fm_ok[i] = (r['tau'] is not None) and verify(inst, r['tau'])['feasible']
            rescue = int((fm_ok & ~cold_ok).sum()); n_coldfail = int((~cold_ok).sum())
            note = "" if apply_fm else "cold-only (M>model max)"
            fm_str = f"{int(fm_ok.sum()):>3}/{a.n}" if apply_fm else "   - "
            resc_str = f"{rescue}/{n_coldfail}" if apply_fm else "  -  "
            print(f"{M:>3} {tau:>6.2f} {int(cold_ok.sum()):>6}/{a.n:<3} {fm_str:>9} "
                  f"{resc_str:>17} {note:>14}")
    print("\ncold_feas/FM_feas = solved at the 2 deg gate; rescue = FM solves among cold's failures.")


if __name__ == '__main__':
    main()
