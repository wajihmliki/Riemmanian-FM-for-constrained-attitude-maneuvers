"""regress_baseline.py -- deterministic regression warm-start baseline.

Same backbone (JointVelocityNetV2) and conditioning as FM-Riem / DDPM-Eucl, but
trained with a direct L2 loss to predict the whole trajectory in one forward
pass (no noise, no reverse process). Deterministic predictors regress toward the
conditional mean, which isolates the benefit of the generative formulation from
the architecture.

    python3 regress_baseline.py --data corridor.pkl --epochs 3000 --out_prefix corridor
"""
import argparse, pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from backbone_v2 import JointVelocityNetV2


class RegressionHead(nn.Module):
    """cond -> full trajectory (B,T,10). Structurally identical to DDPMHead so the
    two baselines differ ONLY in training/sampling, not capacity."""
    def __init__(self, backbone):
        super().__init__(); self.backbone = backbone; self.head = nn.Linear(9, 10)

    def forward(self, x, t_norm, cond):
        return self.head(self.backbone(x, t_norm, cond))


def _sign_align(traj):
    """Canonicalize quaternion double-cover: make scalar part >= 0 so L2 is meaningful."""
    traj = traj.clone()
    q = traj[..., 0:4]
    flip = torch.where(q[..., 0:1] < 0, -1.0, 1.0)
    traj[..., 0:4] = q * flip
    return traj


def train_regress(traj, cond, n_epochs=3000, batch=128, lr=1e-3, hidden=384,
                  n_layers=6, n_heads=8, device='cpu', log_every=300):
    traj = _sign_align(traj.to(device)); cond = cond.to(device)
    N, seq, _ = traj.shape
    m = traj.mean(dim=(0, 1), keepdim=True); s = traj.std(dim=(0, 1), keepdim=True).clamp_min(1e-6)
    x0 = (traj - m) / s
    backbone = JointVelocityNetV2(seq_len=seq, cond_dim=cond.shape[1], hidden=hidden,
                                  n_layers=n_layers, n_heads=n_heads)
    model = RegressionHead(backbone).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    for ep in range(n_epochs):
        idx = torch.randperm(N, device=device)[:batch]
        b = x0[idx]; c = cond[idx]; B = b.size(0)
        xin = torch.zeros(B, seq, 10, device=device)          # dummy input; cond drives it
        t = torch.zeros(B, device=device)
        pred = model(xin, t, c)
        loss = F.mse_loss(pred, b)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if ep % log_every == 0 or ep == n_epochs - 1:
            print(f"  [Regress] ep {ep:4d}  loss = {loss.item():.5f}")
    return model, dict(mean=m.cpu(), std=s.cpu())


@torch.no_grad()
def sample_regress(model, cond, stats, seq_len, device='cpu'):
    model.eval(); cond = cond.to(device); B = cond.size(0)
    mean = stats['mean'].to(device); std = stats['std'].to(device)
    xin = torch.zeros(B, seq_len, 10, device=device); t = torch.zeros(B, device=device)
    out = model(xin, t, cond) * std + mean
    q = out[..., 0:4]; q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    q = torch.where(q[..., :1] < 0, -q, q)
    return torch.cat([q, out[..., 4:10]], dim=-1)


def load_corridor(path):
    d = pickle.load(open(path, "rb"))
    traj = torch.from_numpy(d["trajectories"]).float()
    if traj.shape[1] == 10 and traj.shape[2] != 10:
        traj = traj.transpose(1, 2)
    cond = torch.from_numpy(d["conditions"]).float()
    return traj, cond, d.get("meta", {})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='corridor.pkl'); p.add_argument('--epochs', type=int, default=3000)
    p.add_argument('--batch', type=int, default=128); p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--hidden', type=int, default=384); p.add_argument('--n_layers', type=int, default=6)
    p.add_argument('--n_heads', type=int, default=8)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--out_prefix', default='corridor')
    a = p.parse_args()
    traj, cond, meta = load_corridor(a.data); seq = traj.shape[1]; cond_dim = cond.shape[1]
    print(f"Regression baseline: traj {tuple(traj.shape)}, cond_dim {cond_dim}, device {a.device}")
    model, stats = train_regress(traj, cond, n_epochs=a.epochs, batch=a.batch, lr=a.lr,
                                 hidden=a.hidden, n_layers=a.n_layers, n_heads=a.n_heads,
                                 device=a.device, log_every=max(a.epochs // 10, 1))
    torch.save({'state_dict': model.state_dict(), 'stats': stats, 'seq_len': seq,
                'cond_dim': cond_dim, 'hidden': a.hidden, 'n_layers': a.n_layers,
                'n_heads': a.n_heads, 'kind': 'regress'}, f"{a.out_prefix}_regress.pt")
    print(f"saved {a.out_prefix}_regress.pt")


if __name__ == '__main__':
    main()
