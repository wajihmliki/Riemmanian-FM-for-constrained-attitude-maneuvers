"""eucl_fm.py -- FM-Eucl baseline (quaternion treated as a vector in R^4).

Standard linear flow matching on the full 10-dim state with the same backbone
as FM-Riem. The quaternion is normalized (and sign-canonicalized) only at the
end of sampling.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class EuclHead(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(9, 10)

    def forward(self, x, t, cond):
        return self.head(self.backbone(x, t, cond))


def train_eucl_fm(net, traj_data, cond_data, n_epochs=400, batch_size=64,
                  lr=1e-3, device="cpu", log_every=50):
    traj_data = traj_data.to(device); cond_data = cond_data.to(device)
    N, T, _ = traj_data.shape
    m = traj_data.mean(dim=(0, 1), keepdim=True)
    s = traj_data.std(dim=(0, 1), keepdim=True).clamp_min(1e-6)
    traj_n = (traj_data - m) / s
    model = EuclHead(net).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    for ep in range(n_epochs):
        idx = torch.randperm(N, device=device)[:batch_size]
        x1 = traj_n[idx]; c = cond_data[idx]; B = x1.size(0)
        eps = torch.randn_like(x1)
        t = torch.rand(B, device=device); tb = t.view(B, 1, 1)
        x_t = (1 - tb) * eps + tb * x1
        v_target = x1 - eps
        loss = F.mse_loss(model(x_t, t, c), v_target)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if ep % log_every == 0 or ep == n_epochs - 1:
            print(f"  [Eucl] ep {ep:4d}  loss = {loss.item():.5f}")
    return model, dict(mean=m.cpu(), std=s.cpu())


@torch.no_grad()
def sample_eucl(model, cond, stats, seq_len, n_steps=50, device="cpu"):
    model.eval()
    cond = cond.to(device); B = cond.size(0)
    mean = stats["mean"].to(device); std = stats["std"].to(device)
    z = torch.randn(B, seq_len, 10, device=device)
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t = torch.full((B,), i * dt, device=device)
        z = z + dt * model(z, t, cond)
    out = z * std + mean
    q = out[..., 0:4]
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    q = torch.where(q[..., :1] < 0, -q, q)
    return torch.cat([q, out[..., 4:10]], dim=-1)
