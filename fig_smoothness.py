"""fig_smoothness.py -- Fig. 4: generated-path smoothness.

Mean inter-node geodesic rotation (deg/node) of each method's raw generated
attitude warm start (same measure as diag_warmstart.py), with the SLERP
initialization as the smooth reference.

    python3 fig_smoothness.py --M 5 --tau 0.15 --N 60 --n 60 \
        --fm_riem corridor_fmriem.pt --fm_eucl corridor_fmeucl.pt \
        --ddpm corridor_ddpm.pt --out fig_smoothness.png --device cuda
"""
import argparse, warnings
import numpy as np
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from generate_corridor import make_corridor
from preflight_core import qnorm, qmul, qconj, log_so3
from eval_warmstart import load_fm


def path_smoothness(xi):
    """mean inter-node geodesic rotation (deg) of a (3, N+1) rotation-vector path."""
    return float(np.degrees(np.linalg.norm(np.diff(xi, axis=1), axis=0)).mean())


def slerp_xi(inst):
    """the analytic slerp init as a rotation-vector path (the smooth reference)."""
    q0 = inst['q0']; xt = inst['xi_target']; N = inst['N']
    from preflight_core import exp_so3
    q = [qnorm(qmul(q0, exp_so3((k / N) * xt))) for k in range(N + 1)]
    return np.stack([log_so3(qmul(qconj(q0), q[k])) for k in range(N + 1)], axis=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--M', type=int, default=5); p.add_argument('--tau', type=float, default=0.15)
    p.add_argument('--N', type=int, default=60); p.add_argument('--n', type=int, default=60)
    p.add_argument('--sep', type=float, default=120.0); p.add_argument('--half', type=float, default=20.0)
    p.add_argument('--seed', type=int, default=2025)
    p.add_argument('--fm_riem', default=None); p.add_argument('--fm_eucl', default=None)
    p.add_argument('--ddpm', default=None); p.add_argument('--device', default='cpu')
    p.add_argument('--out', default='fig_smoothness.png')
    a = p.parse_args()

    rng = np.random.default_rng(a.seed)
    insts = [make_corridor(rng, a.M, a.tau, a.N, a.sep, a.half) for _ in range(a.n)]

    arms = {}
    if a.fm_riem: arms['FM-Riem'] = load_fm(a.fm_riem, 'riem', a.device)
    if a.fm_eucl: arms['FM-Eucl'] = load_fm(a.fm_eucl, 'eucl', a.device, reanchor=True)
    if a.ddpm:    arms['DDPM-Eucl'] = load_fm(a.ddpm, 'ddpm', a.device, reanchor=True)

    labels = ['slerp']
    means = [np.mean([path_smoothness(slerp_xi(i)) for i in insts])]
    stds = [np.std([path_smoothness(slerp_xi(i)) for i in insts])]
    for name, fn in arms.items():
        vals = [path_smoothness(fn(i)) for i in insts]
        labels.append(name); means.append(float(np.mean(vals))); stds.append(float(np.std(vals)))
        print(f"{name.splitlines()[0]:>12}: {means[-1]:.1f} +/- {stds[-1]:.1f} deg/node")

    colors = ['#9aa0a6', '#1a73e8', '#e8710a', '#c5221f'][:len(labels)]
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=4, color=colors, edgecolor='black', linewidth=0.6)
    ax.axhline(60, ls='--', color='gray', lw=1)
    ax.text(len(labels) - 1, 62, 'near-random walk (>60°)', ha='right', va='bottom', fontsize=8, color='gray')
    for xi_, m in zip(x, means):
        ax.text(xi_, m + max(stds) * 0.3 + 1, f'{m:.0f}°', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('mean inter-node geodesic rotation (deg/node)', fontsize=9)
    ax.set_title('Generated attitude-path smoothness on SO(3)', fontsize=10)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.tight_layout(); fig.savefig(a.out, dpi=200, bbox_inches='tight')
    print(f"saved {a.out}")


if __name__ == '__main__':
    main()
