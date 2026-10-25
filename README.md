# Riemannian Flow Matching on SO(3) for Constrained Spacecraft Attitude Maneuvers

Code for the paper

> **Riemannian Flow Matching on SO(3) for Constrained Spacecraft Attitude Maneuvers**
> Wajih Imliki and Han-Lim Choi, KAIST.
> Submitted to the 37th AAS/AIAA Astrodynamics Specialist Conference.

A conditional generative model defined intrinsically on the rotation manifold SO(3)
produces warm-start trajectories for successive convexification (SCvx) on cluttered
multi-cone "corridor" attitude maneuvers. Every generated attitude stays on SO(3), the
trajectory is anchored exactly to the commanded boundary attitudes, and every returned
solution is certified by forward-integrating the commanded torque through the full
nonlinear rigid-body dynamics.

Pipeline: data generation → model training → trajectory generation → SCvx warm start →
nonlinear certification → evaluation.

## Installation

Tested with Python 3.10 and PyTorch 2.x (CUDA optional).

```bash
pip install -r requirements.txt
```

`flow_matching` is the [facebookresearch/flow_matching](https://github.com/facebookresearch/flow_matching)
library; `so3_manifold.py` implements its `Manifold` interface for unit quaternions.

All scripts are run from the repository root. Datasets and checkpoints are written to
the working directory.

## Repository layout

| File | Role |
|---|---|
| `so3_manifold.py` | SO(3) manifold (quaternions, body-frame tangents) for `flow_matching` |
| `joint_fm.py` | Riemannian FM on SO(3) × R⁶ (attitude, rate, torque); sampler with boundary anchoring (CVF) |
| `backbone_v2.py` | Transformer backbone shared by every learned method (6 layers, 8 heads, hidden 384) |
| `eucl_fm.py`, `ddpm_eucl.py`, `regress_baseline.py`, `nn_baseline.py` | FM-Eucl, DDPM-Eucl, Regression and NN-Retrieval baselines |
| `preflight_core.py` | SO(3) helpers, true rigid-body dynamics, reference SCvx, and `verify()` — the nonlinear feasibility certificate |
| `fast_scvx.py` | Compile-once (DPP) SCvx used in all experiments |
| `generate_corridor.py`, `generate_corridor_varM.py` | Corridor problems and certified demonstration datasets |
| `train_corridor.py` | Trains FM-Riem and FM-Eucl |
| `eval_warmstart.py` | Feasibility and rescue rate (Table 1) |
| `efficiency_compare.py` | Iterations, runtime and cost on the commonly solved set (Table 2) |
| `difficulty_sweep.py` | Torque and cone-count sweeps (Tables 3–4) |
| `horizon_general.py` | Transfer to longer horizons (Table 5) |
| `fig_smoothness.py`, `diag_warmstart.py` | Generated-path smoothness and iteration-budget study (Fig. 4) |
| `viz_anim.py`, `viz_traj.py` | Sampling progression (Fig. 2) and example maneuvers (Fig. 3) |
| `check_dynamics.py` | Re-integrates any stored dataset under the true dynamics |

## Problem setup

Inertia J = diag(10, 12, 15), torque limit τ_max = 0.15, Δt = 0.6 s, N = 60
(t_f = 36 s), boresight b_body = [0, 0, 1], ~120° start-to-goal separation, M = 5 cones
of ~20° half-angle in an alternating slalom. A trajectory is **feasible** only if the
commanded torque, integrated with RK4 through the full nonlinear dynamics, gives a
terminal attitude error ≤ 2°, a keep-out violation ≤ 0.5°, and a torque within 2% of
the bound.

## Reproducing the paper

### 1. Certified datasets

```bash
python generate_corridor.py --M 5 --tau 0.15 --N 60 --n_unique 2000 --out corridor.pkl
python generate_corridor_varM.py --M_max 10 --tau 0.15 --N 60 --n_unique 3000 --out corridor_varM.pkl
```

Each instance is solved by SCvx from several structured initializations; only the
lowest-cost trajectory that passes the certificate is kept. `check_dynamics.py <pkl>`
re-integrates any dataset under the true dynamics.

### 2. Training (3000 epochs, AdamW lr 1e-3, weight decay 1e-4, cosine schedule, batch 128)

```bash
python train_corridor.py   --data corridor.pkl --out_prefix corridor          # FM-Riem + FM-Eucl
python ddpm_eucl.py        --data corridor.pkl --out_prefix corridor --T 1000 # DDPM-Eucl
python regress_baseline.py --data corridor.pkl --out_prefix corridor          # Regression
python train_corridor.py   --data corridor_varM.pkl --out_prefix corridor_varM # variable-cone model
```

NN-Retrieval needs no training (it reads `corridor.pkl`).

### 3. Results

All evaluations use held-out instances from seed 2025 (Table 1: seeds 2025, 2026, 2027).
Add `--device cuda` to use a GPU.

```bash
# Table 1 -- feasibility and rescue rate (n=500, three seeds)
python eval_warmstart.py --seeds 2025 2026 2027 --n 500 \
    --fm_riem corridor_fmriem.pt --fm_eucl corridor_fmeucl.pt --ddpm corridor_ddpm.pt \
    --regress corridor_regress.pt --nn corridor.pkl

# Table 2 -- iterations, runtime and control cost on the commonly solved set
python efficiency_compare.py --n 120 --fm_riem corridor_fmriem.pt \
    --regress corridor_regress.pt --nn corridor.pkl

# Table 3 -- torque limit (n=80)
python difficulty_sweep.py --M_list 5 --tau_list 0.10 0.12 0.15 0.20 0.30 --n 80 --fm_riem corridor_fmriem.pt

# Table 4 -- number of keep-out cones (n=80)
python difficulty_sweep.py --M_list 1 2 3 4 5 6 7 8 --tau_list 0.15 --n 80 --fm_riem corridor_varM_fmriem.pt

# Table 5 -- longer planning horizons, no retraining
python horizon_general.py --N_list 90 120 --n 80 --fm_riem corridor_fmriem.pt

# Fig. 4 -- generated-path smoothness, and feasibility vs SCvx iteration budget (18/30/50)
python fig_smoothness.py --n 60 --fm_riem corridor_fmriem.pt --fm_eucl corridor_fmeucl.pt --ddpm corridor_ddpm.pt
python diag_warmstart.py --n 60 --fm_riem corridor_fmriem.pt --fm_eucl corridor_fmeucl.pt --ddpm corridor_ddpm.pt

# Robustness -- relaxed 5 deg gate
python eval_warmstart.py --seeds 2025 2026 2027 --n 500 --tol_att 5 \
    --fm_riem corridor_fmriem.pt --fm_eucl corridor_fmeucl.pt --ddpm corridor_ddpm.pt \
    --regress corridor_regress.pt --nn corridor.pkl
python efficiency_compare.py --n 120 --tol_att 5 --fm_riem corridor_fmriem.pt \
    --regress corridor_regress.pt --nn corridor.pkl

# Fig. 2 -- sampling progression (40 integration steps)
python viz_anim.py --anim gen --fm_riem corridor_fmriem.pt --frames 40 --stills 1 15 30 39 40 --out gen_anim.mp4

# Fig. 3 -- one instance, four initializations
python viz_traj.py --mode cold --out fig_cold.png
python viz_traj.py --mode fm --regress corridor_regress.pt --out fig_regress.png
python viz_traj.py --mode fm --nn corridor.pkl --out fig_nn.png
python viz_traj.py --mode fm --fm_riem corridor_fmriem.pt --out fig_fmriem.png
```

The SCvx experiments are CPU-bound; the 500-instance runs take several hours.

## Citation

Citation details will be added once the paper is published.

## Acknowledgements

Supported in part by the Korea AeroSpace Administration (KASA) under Grant
2022M1A3C2069728, Future Space Education Center.

## License

MIT — see [LICENSE](LICENSE).
