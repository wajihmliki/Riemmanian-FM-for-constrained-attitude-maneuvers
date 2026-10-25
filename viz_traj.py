"""viz_traj.py -- Fig. 3: one maneuver against its keep-out cones.

Produces a 2x2 figure:
  (1) boresight path on the unit sphere with the keep-out cone boundaries,
      start = green, target = red
  (2) attitude error to target vs time
  (3) body rate magnitude vs time
  (4) commanded torque per axis vs time, with the +/- limit

Trajectory source (--mode):
  cold  : solve the instance with SCvx from the SLERP init (--seed/--idx pick the instance)
  fm    : warm-start SCvx from a learned model (--fm_riem | --regress | --nn ...)
All modes forward-integrate the true nonlinear dynamics before plotting and print
the certificate verdict.

    python3 viz_traj.py --mode cold --seed 9999 --idx 0 --out cold.png
    python3 viz_traj.py --mode fm --fm_riem corridor_fmriem.pt --seed 9999 --idx 0 --out fmriem.png
    python3 viz_traj.py --mode fm --regress corridor_regress.pt --seed 9999 --idx 0 --out regress.png
    python3 viz_traj.py --mode fm --nn corridor.pkl --seed 9999 --idx 0 --out nn.png
"""
import argparse, warnings
import numpy as np
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from preflight_core import (integrate_true, log_so3, exp_so3, qmul, qconj, qnorm, rotmat, verify)
from fast_scvx import FastSCvx
from generate_corridor import make_corridor


def boresight(qs, b):
    return np.array([rotmat(qs[:, k]) @ b for k in range(qs.shape[1])])


def cone_circle(cdir, exc_deg, npts=90):
    cdir = cdir / np.linalg.norm(cdir); exc = np.radians(exc_deg)
    a = np.array([1., 0, 0]) if abs(cdir[0]) < 0.9 else np.array([0, 1., 0])
    u = np.cross(cdir, a); u /= np.linalg.norm(u); v = np.cross(cdir, u)
    th = np.linspace(0, 2 * np.pi, npts)
    return np.array([np.cos(exc) * cdir + np.sin(exc) * (np.cos(t) * u + np.sin(t) * v) for t in th])


def get_tau(args, inst):
    S = FastSCvx(N=inst["N"], ncone=len(inst["cones"]), dt=0.6,
                 tau_max=inst["tau_max"], J=inst["J"])
    xi0 = None
    if args.mode == "fm":
        from eval_warmstart import load_fm
        if args.nn:
            from nn_baseline import make_nn_init
            fn = make_nn_init(args.nn, reanchor=True)
        else:
            if args.ddpm:
                ck, kind = args.ddpm, "ddpm"
            elif args.regress:
                ck, kind = args.regress, "regress"
            elif args.fm_eucl:
                ck, kind = args.fm_eucl, "eucl"
            else:
                ck, kind = args.fm_riem, "riem"
            fn = load_fm(ck, kind, args.device, reanchor=(kind != "riem"))
        xi0 = fn(inst)
    r = S.solve(inst, xi_init=xi0, n_iter=18, trust_weight=8.0)
    return r["tau"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["cold", "fm"], default="cold")
    p.add_argument("--idx", type=int, default=0)
    p.add_argument("--M", type=int, default=5); p.add_argument("--tau", type=float, default=0.15)
    p.add_argument("--N", type=int, default=60); p.add_argument("--sep", type=float, default=120.0)
    p.add_argument("--half", type=float, default=20.0); p.add_argument("--seed", type=int, default=9999)
    p.add_argument("--fm_riem", default=None); p.add_argument("--fm_eucl", default=None)
    p.add_argument("--ddpm", default=None)
    p.add_argument("--regress", default=None); p.add_argument("--nn", default=None)
    p.add_argument("--device", default="cpu"); p.add_argument("--out", default="traj.png")
    a = p.parse_args()

    rng = np.random.default_rng(a.seed)
    inst = None
    for _ in range(a.idx + 1):
        inst = make_corridor(rng, a.M, a.tau, a.N, a.sep, a.half)   # idx-th instance from seed
    tau = get_tau(a, inst)
    if tau is None:
        print("solve failed (no trajectory to plot)"); return
    qs, ws = integrate_true(inst["q0"], inst["w0"], tau, inst["dt"], inst["J"], substeps=15)
    v = verify(inst, tau)
    print(f"verifier: feasible={v['feasible']}  att_err={v['att_err']:.2f}°  "
          f"max_vio={v['max_vio']:.2f}°  max_tau={v['max_tau']:.3f}")

    b = inst["b_body"]; bs = boresight(qs, b)
    bstart = rotmat(inst["q0"]) @ b
    btgt = rotmat(inst["q_tgt"]) @ b
    dt = inst["dt"]; T = qs.shape[1]; tt = np.arange(T) * dt
    att = np.array([np.degrees(np.linalg.norm(log_so3(qmul(qconj(qs[:, k]), inst["q_tgt"])))) for k in range(T)])
    rate = np.linalg.norm(ws, axis=0)

    fig = plt.figure(figsize=(13, 9))
    ax = fig.add_subplot(2, 2, 1, projection="3d")
    # faint sphere
    u, vv = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
    ax.plot_wireframe(np.cos(u)*np.sin(vv), np.sin(u)*np.sin(vv), np.cos(vv),
                      color="lightgray", lw=0.3, alpha=0.5)
    for (cdir, cexc) in inst["cones"]:
        exc = np.degrees(np.arccos(np.clip(cexc, -1, 1)))
        c = cone_circle(np.asarray(cdir), exc)
        ax.plot(c[:, 0], c[:, 1], c[:, 2], color="crimson", lw=1.6, alpha=0.85)
    ax.plot(bs[:, 0], bs[:, 1], bs[:, 2], color="navy", lw=2.4, label="boresight path")
    ax.scatter(*bstart, color="green", s=60, label="start")
    ax.scatter(*btgt, color="red", s=60, label="target")
    ax.set_title(f"boresight vs keep-out cones  (feasible={v['feasible']})")
    ax.legend(loc="upper left", fontsize=8); ax.set_box_aspect((1, 1, 1))

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(tt, att, color="navy"); ax2.axhline(2, ls="--", color="gray", lw=1)
    ax2.set_title("attitude error to target"); ax2.set_xlabel("time (s)"); ax2.set_ylabel("deg")

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(tt, rate, color="darkorange")
    ax3.set_title("body rate magnitude"); ax3.set_xlabel("time (s)"); ax3.set_ylabel("rad/s")

    ax4 = fig.add_subplot(2, 2, 4)
    tw = np.arange(tau.shape[1]) * dt
    for i, lab in enumerate(["tau_x", "tau_y", "tau_z"]):
        ax4.plot(tw, tau[i], label=lab)
    ax4.axhline(inst["tau_max"], ls="--", color="gray", lw=1)
    ax4.axhline(-inst["tau_max"], ls="--", color="gray", lw=1)
    ax4.set_title("commanded torque"); ax4.set_xlabel("time (s)"); ax4.set_ylabel("N·m")
    ax4.legend(fontsize=8)

    fig.suptitle(f"mode={a.mode}  seed={a.seed}  idx={a.idx}  M={a.M}  tau={a.tau}", fontsize=11)
    fig.tight_layout(); fig.savefig(a.out, dpi=120)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
