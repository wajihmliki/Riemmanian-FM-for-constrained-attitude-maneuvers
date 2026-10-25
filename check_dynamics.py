"""check_dynamics.py -- re-integrate an existing dataset under the true dynamics.

Loads a .pkl of attitude demonstrations, takes each stored (q, omega, tau)
trajectory, forward-integrates the COMMANDED torque under the full nonlinear
rigid-body dynamics, and reports how far the real endpoint lands from the stored
endpoint and whether the keep-out constraints hold. Use it to check whether a
dataset produced with optimizer termination criteria alone is dynamically valid.

    python3 check_dynamics.py some_dataset.pkl --n 300
"""
import argparse, pickle
import numpy as np
from preflight_core import integrate_true, log_so3, qmul, qconj, qnorm, rotmat


def get_mat(traj):
    """Return a (10, T) matrix [q(4); om(3); tau(3)] regardless of stored axis order."""
    traj = np.asarray(traj)
    if traj.shape[0] == 10:
        return traj
    if traj.shape[1] == 10:
        return traj.T
    raise ValueError(f"unexpected trajectory shape {traj.shape}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data")
    p.add_argument("--n", type=int, default=200, help="number of trajectories to check")
    p.add_argument("--substeps", type=int, default=15)
    p.add_argument("--tol_att", type=float, default=2.0, help="terminal-attitude gate (deg)")
    args = p.parse_args()

    d = pickle.load(open(args.data, "rb"))
    meta = d.get("meta", {})
    J = np.array(meta.get("J", [[10, 0, 0], [0, 12, 0], [0, 0, 15]]), dtype=float)
    if J.ndim == 1:
        J = np.diag(J)
    b_body = np.array(meta.get("b_body", [0, 0, 1.0]), dtype=float)
    N = int(meta.get("N", get_mat(d["trajectories"][0]).shape[1] - 1))
    tf = float(meta.get("tf", N * 0.6))
    dt = tf / N
    cone_dirs = [np.array(x, dtype=float) for x in meta.get("cone_dirs", [])]
    cone_half = meta.get("cone_half_deg", [])
    # corridor-style datasets store the cones per instance in the conditioning:
    # q0(4) q_tgt(4) cone_dirs(3M) cos_excs(M)
    per_inst = not cone_dirs and "conditions" in d
    src = "per-instance (conditions)" if per_inst else str(len(cone_dirs))
    print(f"{args.data}: N={N} tf={tf} dt={dt:.3f}  cones={src}  J=diag{np.diag(J).tolist()}")

    trajs = d["trajectories"]
    idx = np.random.default_rng(0).choice(len(trajs), min(args.n, len(trajs)), replace=False)
    att_errs, vios, feas = [], [], 0
    for i in idx:
        M = get_mat(trajs[i])
        q = M[0:4]; om = M[4:7]; tau = M[7:10][:, :N]   # drop padded last col
        q0 = qnorm(q[:, 0]); w0 = om[:, 0]
        qs, ws = integrate_true(q0, w0, tau, dt, J, substeps=args.substeps)
        att = np.degrees(np.linalg.norm(log_so3(qmul(qconj(qnorm(qs[:, -1])), qnorm(q[:, -1])))))
        att_errs.append(att)
        if per_inst:
            c = np.asarray(d["conditions"][i], dtype=float); M_c = (c.size - 8) // 4
            dirs = c[8:8 + 3 * M_c].reshape(M_c, 3); coss = c[8 + 3 * M_c:8 + 4 * M_c]
            act = coss < 1.0                                   # skip padded (inactive) cones
            cone_dirs = list(dirs[act])
            cone_half = list(np.degrees(np.arccos(np.clip(coss[act], -1, 1))))
        mv = 0.0
        for cdir, ch in zip(cone_dirs, cone_half):
            for k in range(qs.shape[1]):
                ang = np.degrees(np.arccos(np.clip((rotmat(qs[:, k]) @ b_body) @ cdir, -1, 1)))
                mv = max(mv, ch - ang)
        mv = max(0.0, mv); vios.append(mv)
        if att < args.tol_att and mv < 0.5:
            feas += 1
    a = np.array(att_errs); v = np.array(vios)
    print(f"\nTerminal attitude error under TRUE dynamics (n={len(idx)}):")
    print(f"  median {np.median(a):.2f}deg   p90 {np.percentile(a,90):.2f}deg   max {a.max():.2f}deg")
    if cone_dirs:
        print(f"keep-out violation under TRUE dynamics: median {np.median(v):.2f}deg  max {v.max():.2f}deg")
    print(f"physically feasible (<{args.tol_att:g}deg terminal AND <0.5deg violation): "
          f"{feas}/{len(idx)} = {feas/len(idx):.0%}")


if __name__ == "__main__":
    main()
