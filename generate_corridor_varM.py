"""generate_corridor_varM.py -- one certified dataset covering variable cone counts.

Each problem has a random number of ACTIVE cones (M_active in [M_min, M_max]),
padded to M_max=10 INACTIVE cones (dir=[0,0,1], cos_exc=1.0 -> zero-size
forbidden region). Conditioning is therefore fixed at cond_dim = 8 + 4*M_max = 48,
so a single FM trained here covers the whole cone-count sweep (Table 4). Demos come
from the structured demonstrator + nonlinear certificate (2 deg gate). Per-M
feasibility is printed.

Cond layout: q0(4) q_tgt(4) cone_dirs(3*M_max) cos_excs(M_max), active first.

    python3 generate_corridor_varM.py --M_max 10 --tau 0.15 --N 60 \
        --n_unique 3000 --out corridor_varM.pkl
"""
import argparse, pickle, warnings
import numpy as np
warnings.filterwarnings("ignore")
from tqdm import tqdm
from generate_corridor import rand_rot, quat_z_to, struct_best
from preflight_core import qnorm, qmul, qconj, exp_so3, log_so3
from fast_scvx import FastSCvx

PAD_DIR = np.array([0., 0., 1.])      # inactive-cone direction (cos_exc=1.0 -> no-op)


def make_varM(rng, M_active, M_max, tau, N, sep_deg=120.0, half_deg=20.0):
    R = rand_rot(rng)
    bs0 = R @ np.array([np.cos(np.deg2rad(-sep_deg/2)), 0, np.sin(np.deg2rad(-sep_deg/2))])
    bsT = R @ np.array([np.cos(np.deg2rad( sep_deg/2)), 0, np.sin(np.deg2rad( sep_deg/2))])
    bs0 /= np.linalg.norm(bs0); bsT /= np.linalg.norm(bsT)
    lateral = np.cross(bs0, bsT); lateral /= np.linalg.norm(lateral)
    perp = np.cross(np.cross(bs0, bsT), bs0); perp /= np.linalg.norm(perp)
    cones = []
    for m in range(M_active):
        frac = (m+1)/(M_active+1)
        mid = bs0*np.cos(frac*np.deg2rad(sep_deg)) + perp*np.sin(frac*np.deg2rad(sep_deg))
        off = (-1)**m * np.deg2rad(rng.uniform(7, 14))
        cdir = mid*np.cos(off) + lateral*np.sin(off); cdir /= np.linalg.norm(cdir)
        cones.append((cdir, np.cos(np.deg2rad(half_deg))))
    for _ in range(M_max - M_active):
        cones.append((PAD_DIR.copy(), 1.0))                  # inactive
    q0 = qnorm(qmul(quat_z_to(bs0), exp_so3(np.array([0, 0, rng.uniform(-0.3, 0.3)]))))
    q_tgt = qnorm(qmul(quat_z_to(bsT), exp_so3(np.array([0, 0, rng.uniform(-0.3, 0.3)]))))
    xt = log_so3(qmul(qconj(q0), q_tgt))
    return dict(J=np.diag([10., 12., 15.]), dt=0.6, N=N, b_body=np.array([0., 0, 1.]),
                tau_max=tau, w0=np.zeros(3), xi_target=xt, q0=q0, q_tgt=q_tgt,
                cones=cones, M_active=M_active)


def cond_of(inst):
    dirs = np.concatenate([c[0] for c in inst['cones']])
    cos = np.array([c[1] for c in inst['cones']])
    return np.concatenate([inst['q0'], inst['q_tgt'], dirs, cos]).astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--M_max', type=int, default=10); p.add_argument('--M_min', type=int, default=1)
    p.add_argument('--tau', type=float, default=0.15); p.add_argument('--N', type=int, default=60)
    p.add_argument('--sep', type=float, default=120.0); p.add_argument('--half', type=float, default=20.0)
    p.add_argument('--n_unique', type=int, default=3000)
    p.add_argument('--max_attempts', type=int, default=20000)
    p.add_argument('--seed', type=int, default=11)
    p.add_argument('--out', default='corridor_varM.pkl')
    a = p.parse_args()

    S = FastSCvx(N=a.N, ncone=a.M_max, dt=0.6, tau_max=a.tau, J=np.diag([10., 12., 15.]))
    rng = np.random.default_rng(a.seed)
    trajs, conds, mact = [], [], []
    kept = {m: 0 for m in range(a.M_min, a.M_max+1)}
    tried = {m: 0 for m in range(a.M_min, a.M_max+1)}
    pbar = tqdm(total=a.n_unique, desc='varM')
    attempts = 0
    while len(conds) < a.n_unique and attempts < a.max_attempts:
        attempts += 1
        M_active = int(rng.integers(a.M_min, a.M_max + 1))
        tried[M_active] += 1
        inst = make_varM(rng, M_active, a.M_max, a.tau, a.N, a.sep, a.half)
        sol = struct_best(S, inst)               # (cost, xi, om, tau)
        if sol is None:
            continue
        _, xi, om, tau = sol
        quats = _xi_to_quat(xi, inst['q0'])                       # (4, N+1)
        tau_pad = np.concatenate([tau, tau[:, -1:]], axis=1)      # (3, N) -> (3, N+1)
        traj = np.concatenate([quats, om, tau_pad], axis=0)       # (10, N+1)
        trajs.append(traj.astype(np.float32)); conds.append(cond_of(inst)); mact.append(M_active)
        kept[M_active] += 1; pbar.update(1)
    pbar.close()

    data = dict(trajectories=np.stack(trajs), conditions=np.stack(conds),
                M_active=np.array(mact),
                meta=dict(M_max=a.M_max, tau=a.tau, N=a.N,
                          cond_layout='q0(4) q_tgt(4) cone_dirs(3*M_max) cos_excs(M_max)'))
    pickle.dump(data, open(a.out, 'wb'))
    print(f"\nSaved {len(trajs)} demos -> {a.out}  (overall feasibility {len(trajs)/attempts:.1%})")
    print("per-M kept / tried (the learnable range):")
    for m in range(a.M_min, a.M_max+1):
        rate = kept[m]/tried[m] if tried[m] else 0.0
        print(f"  M={m:2d}: kept {kept[m]:4d} / tried {tried[m]:4d}  ({rate:.0%})")


def _xi_to_quat(xi, q0):
    """xi (3,T) relative to q0 -> absolute quaternion channels (4,T)."""
    T = xi.shape[1]
    q = np.stack([qnorm(qmul(q0, exp_so3(xi[:, k]))) for k in range(T)], axis=1)
    return q


if __name__ == '__main__':
    main()
