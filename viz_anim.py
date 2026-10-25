"""viz_anim.py -- Fig. 2: FM-Riem sampling progression, and a maneuver animation.

Two modes:
  --anim gen    : the generation process. The boresight path unfolds from noise to a
                  smooth, cone-avoiding route on the sphere as the flow is integrated
                  (40 integration steps by default; --frames).
  --anim solve  : the refined maneuver playing out in time (boresight, attitude
                  error, body rate).

Saves MP4 if ffmpeg is available, else GIF. --stills k1 k2 ... additionally saves
PNG frames at those steps (e.g. 1 15 30 39 40 for Fig. 2).

    python3 viz_anim.py --anim gen --fm_riem corridor_fmriem.pt --seed 42 --idx 1 \
        --M 5 --tau 0.15 --N 60 --out gen_anim.mp4 --stills 1 15 30 39 40 --device cuda
    python3 viz_anim.py --anim solve --fm_riem corridor_fmriem.pt --seed 42 --idx 1 \
        --M 5 --tau 0.15 --N 60 --out solve_anim.mp4 --device cuda
"""
import argparse, warnings
import numpy as np
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D  # noqa
from preflight_core import qnorm, qmul, qconj, exp_so3, log_so3, verify, rotmat
from fast_scvx import FastSCvx
from generate_corridor import make_corridor
from viz_traj import boresight as _boresight, cone_circle as _cone_circle


def boresight_of(q, b_body):
    """rotate body boresight by quaternion q -> inertial unit vector."""
    return rotmat(q) @ b_body


def draw_sphere(ax):
    u, v = np.mgrid[0:2 * np.pi:24j, 0:np.pi:12j]
    ax.plot_wireframe(np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v),
                      color="lightgray", lw=0.3, alpha=0.5)
    ax.set_box_aspect([1, 1, 1]); ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)


def quats_from_xi(xi, q0):
    return [qnorm(qmul(q0, exp_so3(xi[:, k]))) for k in range(xi.shape[1])]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--anim', choices=['gen', 'solve'], default='gen')
    p.add_argument('--fm_riem', required=True); p.add_argument('--seed', type=int, default=42)
    p.add_argument('--idx', type=int, default=1); p.add_argument('--M', type=int, default=5)
    p.add_argument('--tau', type=float, default=0.15); p.add_argument('--N', type=int, default=60)
    p.add_argument('--sep', type=float, default=120.0); p.add_argument('--half', type=float, default=20.0)
    p.add_argument('--frames', type=int, default=40); p.add_argument('--fps', type=int, default=12)
    p.add_argument('--out', default='anim.mp4'); p.add_argument('--device', default='cpu')
    p.add_argument('--stills', type=int, nargs='*', default=None,
                   help='(gen) also save PNG frames at these 1-based flow steps')
    a = p.parse_args()

    rng = np.random.default_rng(a.seed)
    insts = [make_corridor(rng, a.M, a.tau, a.N, a.sep, a.half) for _ in range(a.idx + 1)]
    inst = insts[a.idx]
    b_body = inst['b_body']

    if a.anim == 'gen':
        # integrate the FM ODE, snapshotting the boresight path at each flow step
        from joint_fm import sample_joint_trace  # returns list of (T,10) states along flow
        from eval_warmstart import make_cond
        import torch
        from backbone_v2 import JointVelocityNetV2
        ck = torch.load(a.fm_riem, map_location=a.device); T = ck['seq_len']
        m_max = (ck['cond_dim'] - 8) // 4
        net = JointVelocityNetV2(seq_len=T, cond_dim=ck['cond_dim'], hidden=ck['hidden'],
                                 n_layers=ck['n_layers'], n_heads=ck['n_heads'])
        net.load_state_dict(ck['state_dict']); net.to(a.device).eval()
        cond = make_cond(inst, m_max=m_max).to(a.device)
        trace = sample_joint_trace(net, cond, ck['stats'], seq_len=T, n_steps=a.frames,
                                   device=a.device)  # list of (T,10)
        paths = []
        for st in trace:
            q = st[:, 0:4]
            for k in range(1, len(q)):
                if np.dot(q[k], q[k - 1]) < 0: q[k] = -q[k]
            base = qnorm(q[0]); qs = [qnorm(qmul(base, qnorm(qmul(qconj(base), qnorm(q[k]))))) for k in range(len(q))]
            paths.append(np.array([boresight_of(qq, b_body) for qq in qs]).T)

        fig = plt.figure(figsize=(7, 6)); ax = fig.add_subplot(111, projection='3d')
        def upd(i):
            ax.clear(); draw_sphere(ax)
            for (cd, ce) in inst['cones']:
                if ce >= 1.0: continue
                cc = _cone_circle(cd, np.rad2deg(np.arccos(ce))).T; ax.plot(*cc, color="red", lw=1, alpha=0.7)
            P = paths[i]; ax.plot(*P, color="navy", lw=2)
            ax.scatter(*P[:, 0], color="green", s=40); ax.scatter(*P[:, -1], color="red", s=40)
            ax.set_title(f"FM generation  step {i+1}/{len(paths)}  (noise -> route)")
        ani = animation.FuncAnimation(fig, upd, frames=len(paths), interval=1000 // a.fps)
        for k in (a.stills or []):
            upd(k - 1)
            fn = a.out.rsplit('.', 1)[0] + f'_step{k:02d}.png'
            fig.savefig(fn, dpi=150, bbox_inches='tight'); print(f"saved {fn}")

    else:  # solve: the maneuver playing out (animated 4-panel)
        from eval_warmstart import load_fm
        fn = load_fm(a.fm_riem, 'riem', a.device)
        S = FastSCvx(N=a.N, ncone=len(inst['cones']), dt=0.6, tau_max=inst['tau_max'], J=inst['J'])
        r = S.solve(inst, xi_init=fn(inst), n_iter=18, trust_weight=8.0)
        tau = r['tau']
        if tau is None:
            raise SystemExit("SCvx returned no solution for this instance")
        v = verify(inst, tau)
        # plot the motion the commanded torque actually produces (true dynamics)
        from preflight_core import integrate_true
        qs_t, om = integrate_true(inst['q0'], inst['w0'], tau, inst['dt'], inst['J'], substeps=15)
        qs = qs_t.T

        tt = np.arange(len(qs)) * inst['dt']
        bores = np.array([boresight_of(q, b_body) for q in qs]).T
        att = np.array([np.rad2deg(np.linalg.norm(log_so3(qmul(qconj(q), inst['q_tgt'])))) for q in qs])
        rate = np.linalg.norm(om, axis=0)

        fig = plt.figure(figsize=(12, 5))
        ax1 = fig.add_subplot(131, projection='3d'); ax2 = fig.add_subplot(132); ax3 = fig.add_subplot(133)
        def upd(i):
            ax1.clear(); draw_sphere(ax1)
            for (cd, ce) in inst['cones']:
                if ce >= 1.0: continue
                cc = _cone_circle(cd, np.rad2deg(np.arccos(ce))).T; ax1.plot(*cc, color="red", lw=1, alpha=0.7)
            ax1.plot(*bores[:, :i + 1], color="navy", lw=2)
            ax1.scatter(*bores[:, 0], color="green", s=30); ax1.scatter(*bores[:, -1], color="red", s=30)
            ax1.set_title(f"boresight  (feasible={v['feasible']})")
            ax2.clear(); ax2.plot(tt[:i + 1], att[:i + 1], color="navy"); ax2.axhline(2, ls="--", color="gray", lw=1)
            ax2.set_xlim(0, tt[-1]); ax2.set_ylim(0, max(att) * 1.05); ax2.set_title("attitude error (deg)")
            ax3.clear(); ax3.plot(tt[:i + 1], rate[:i + 1], color="darkorange")
            ax3.set_xlim(0, tt[-1]); ax3.set_ylim(0, max(rate) * 1.05); ax3.set_title("body rate (rad/s)")
        ani = animation.FuncAnimation(fig, upd, frames=len(qs), interval=1000 // a.fps)

    try:
        ani.save(a.out, writer=animation.FFMpegWriter(fps=a.fps))
        print(f"saved {a.out} (mp4)")
    except Exception:
        alt = a.out.rsplit('.', 1)[0] + '.gif'
        ani.save(alt, writer=animation.PillowWriter(fps=a.fps))
        print(f"ffmpeg unavailable; saved {alt} (gif)")


if __name__ == '__main__':
    main()
