"""train_corridor.py -- train FM-Riem and FM-Eucl on certified corridor demonstrations.

    python3 train_corridor.py --data corridor.pkl --epochs 3000 --hidden 384 --out_prefix corridor
    python3 train_corridor.py --data corridor_varM.pkl --epochs 3000 --hidden 384 --out_prefix corridor_varM

Saves <prefix>_fmriem.pt and <prefix>_fmeucl.pt (AdamW, lr 1e-3, weight decay 1e-4,
cosine schedule, batch 128).
"""
import argparse, pickle
import numpy as np
import torch
from backbone_v2 import JointVelocityNetV2
from joint_fm import train_joint_fm
from eucl_fm import train_eucl_fm


def load_corridor(path):
    d = pickle.load(open(path, "rb"))
    traj = torch.from_numpy(d["trajectories"]).float()
    if traj.shape[1] == 10 and traj.shape[2] != 10:
        traj = traj.transpose(1, 2)              # -> (N, T, 10)
    cond = torch.from_numpy(d["conditions"]).float()
    return traj, cond, d["meta"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="corridor.pkl")
    p.add_argument("--epochs", type=int, default=3000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=384)
    p.add_argument("--n_layers", type=int, default=6)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out_prefix", default="corridor")
    args = p.parse_args()

    traj, cond, meta = load_corridor(args.data)
    seq_len = traj.shape[1]; cond_dim = cond.shape[1]
    print(f"corridor: traj {tuple(traj.shape)}, cond_dim {cond_dim}, device {args.device}")

    def save(model, stats, tag):
        torch.save({"state_dict": model.state_dict(), "stats": stats,
                    "seq_len": seq_len, "cond_dim": cond_dim, "hidden": args.hidden,
                    "n_layers": args.n_layers, "n_heads": args.n_heads,
                    "M_cones": meta.get("M_cones")},
                   f"{args.out_prefix}_{tag}.pt")
        print(f"saved {args.out_prefix}_{tag}.pt")

    print("\n--- FM-Riem ---")
    br = JointVelocityNetV2(seq_len=seq_len, cond_dim=cond_dim, hidden=args.hidden,
                            n_layers=args.n_layers, n_heads=args.n_heads)
    rm, rst = train_joint_fm(br, traj, cond, n_epochs=args.epochs, batch_size=args.batch,
                             lr=args.lr, device=args.device, log_every=max(args.epochs // 10, 1))
    save(rm, rst, "fmriem")

    print("\n--- FM-Eucl ---")
    be = JointVelocityNetV2(seq_len=seq_len, cond_dim=cond_dim, hidden=args.hidden,
                            n_layers=args.n_layers, n_heads=args.n_heads)
    em, est = train_eucl_fm(be, traj, cond, n_epochs=args.epochs, batch_size=args.batch,
                            lr=args.lr, device=args.device, log_every=max(args.epochs // 10, 1))
    save(em, est, "fmeucl")


if __name__ == "__main__":
    main()
