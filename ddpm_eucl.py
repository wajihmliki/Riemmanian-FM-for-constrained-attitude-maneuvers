"""ddpm_eucl.py -- DDPM-Eucl baseline (diffusion in the Euclidean quaternion embedding).

A controlled diffusion baseline on the same corridor data, backbone and nonlinear
feasibility certificate as FM-Riem. It is not a reproduction of TrajDiffuser; it
uses the parts that define the family:
  - denoising diffusion (DDPM) with a cosine noise schedule and eps-prediction,
  - the quaternion treated as raw R^4, normalized only at the end,
  - the generated trajectory used to warm-start SCvx.

The network is JointVelocityNetV2 with a 10-dim eps head; sampling uses the full
T=1000-step ancestral chain.

    python3 ddpm_eucl.py --data corridor.pkl --epochs 3000 --T 1000 --out_prefix corridor
"""
import argparse, pickle, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from backbone_v2 import JointVelocityNetV2


# ---------- cosine schedule (Nichol & Dhariwal, as used by TrajDiffuser) -------
def cosine_alpha_bar(T, s=0.008):
    t = torch.linspace(0, T, T + 1)
    f = torch.cos(((t / T + s) / (1 + s)) * math.pi / 2) ** 2
    ab = f / f[0]
    betas = (1 - ab[1:] / ab[:-1]).clamp(1e-6, 0.999)
    return betas


class DDPMHead(nn.Module):
    """Backbone (B,T,9) -> eps over the full 10-dim state."""
    def __init__(self, backbone):
        super().__init__(); self.backbone = backbone; self.head = nn.Linear(9, 10)

    def forward(self, x, t_norm, cond):
        return self.head(self.backbone(x, t_norm, cond))


def train_ddpm(traj, cond, n_epochs=3000, T=1000, batch=128, lr=1e-3,
               hidden=384, n_layers=6, n_heads=8, device='cpu', log_every=300):
    """traj: (N, seq, 10) raw (quat unit, omega/tau raw). cond: (N, cond_dim)."""
    traj = traj.to(device); cond = cond.to(device)
    N, seq, _ = traj.shape
    m = traj.mean(dim=(0, 1), keepdim=True)
    s = traj.std(dim=(0, 1), keepdim=True).clamp_min(1e-6)
    x0 = (traj - m) / s                                   # robust-ish standardization

    betas = cosine_alpha_bar(T).to(device)
    alphas = 1.0 - betas
    abar = torch.cumprod(alphas, dim=0)                   # (T,)

    backbone = JointVelocityNetV2(seq_len=seq, cond_dim=cond.shape[1], hidden=hidden,
                                  n_layers=n_layers, n_heads=n_heads)
    model = DDPMHead(backbone).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

    for ep in range(n_epochs):
        idx = torch.randperm(N, device=device)[:batch]
        b = x0[idx]; c = cond[idx]; B = b.size(0)
        t = torch.randint(0, T, (B,), device=device)
        ab = abar[t].view(B, 1, 1)
        eps = torch.randn_like(b)
        xt = torch.sqrt(ab) * b + torch.sqrt(1 - ab) * eps
        t_norm = (t.float() / T)
        eps_pred = model(xt, t_norm, c)
        loss = F.mse_loss(eps_pred, eps)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if ep % log_every == 0 or ep == n_epochs - 1:
            print(f"  [DDPM] ep {ep:4d}  loss = {loss.item():.5f}")
    stats = dict(mean=m.cpu(), std=s.cpu(), betas=betas.cpu(), abar=abar.cpu(), T=T)
    return model, stats


@torch.no_grad()
def sample_ddpm(model, cond, stats, seq_len, device='cpu'):
    model.eval(); cond = cond.to(device); B = cond.size(0)
    mean = stats['mean'].to(device); std = stats['std'].to(device)
    betas = stats['betas'].to(device); abar = stats['abar'].to(device); T = stats['T']
    alphas = 1.0 - betas
    x = torch.randn(B, seq_len, 10, device=device)
    for t in reversed(range(T)):
        t_b = torch.full((B,), t, device=device)
        ab = abar[t]; a = alphas[t]
        eps = model(x, (t_b.float() / T), cond)
        mean_t = (x - betas[t] / torch.sqrt(1 - ab) * eps) / torch.sqrt(a)
        if t > 0:
            ab_prev = abar[t-1]
            var = betas[t] * (1 - ab_prev) / (1 - ab)
            x = mean_t + torch.sqrt(var.clamp_min(1e-20)) * torch.randn_like(x)
        else:
            x = mean_t
    out = x * std + mean
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
    p.add_argument('--data', default='corridor.pkl')
    p.add_argument('--epochs', type=int, default=3000); p.add_argument('--T', type=int, default=1000)
    p.add_argument('--batch', type=int, default=128); p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--hidden', type=int, default=384); p.add_argument('--n_layers', type=int, default=6)
    p.add_argument('--n_heads', type=int, default=8)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--out_prefix', default='corridor')
    a = p.parse_args()
    traj, cond, meta = load_corridor(a.data)
    seq = traj.shape[1]; cond_dim = cond.shape[1]
    print(f"DDPM-Eucl: traj {tuple(traj.shape)}, cond_dim {cond_dim}, T={a.T}, device {a.device}")
    model, stats = train_ddpm(traj, cond, n_epochs=a.epochs, T=a.T, batch=a.batch, lr=a.lr,
                              hidden=a.hidden, n_layers=a.n_layers, n_heads=a.n_heads,
                              device=a.device, log_every=max(a.epochs // 10, 1))
    torch.save({'state_dict': model.state_dict(), 'stats': stats, 'seq_len': seq,
                'cond_dim': cond_dim, 'hidden': a.hidden, 'n_layers': a.n_layers,
                'n_heads': a.n_heads, 'kind': 'ddpm'},
               f"{a.out_prefix}_ddpm.pt")
    print(f"saved {a.out_prefix}_ddpm.pt")


if __name__ == '__main__':
    main()
