"""Joint SO(3) x R^6 flow matching for full attitude state trajectories.

State: (q, omega, tau) where
  q   ∈ SO(3) (quaternion, 4 components)
  omega ∈ R^3 (angular velocity)
  tau   ∈ R^3 (torque)

The trajectory is stored as (B, 10, T) with channel layout:
  0:4  quaternion
  4:7  omega
  7:10 tau

Training:
  - attitude channels: Riemannian FM using the library's GeodesicProbPath +
    quaternion-rate-to-body-velocity target
  - omega/tau channels: standard linear FM (target = x_1 - x_0)
Both losses are mean-squared error against per-channel targets, summed.

Sampling:
  - attitude: integrate on SO(3) via expmap with body velocity
  - omega/tau: standard Euler

Includes manifold-aware boundary anchoring (CVF) at sampling time:
  - initial attitude q(0) -> q_init (from the condition)
  - final attitude   q(T) -> q_goal (from the condition)
  - boundary rates   omega(0) = omega(T) = 0
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from flow_matching.path import GeodesicProbPath
from flow_matching.path.scheduler import CondOTScheduler
from so3_manifold import SO3, _so3_exp, _quat_mul, _quat_conj


class JointVelocityNet(nn.Module):
    """Transformer that predicts joint velocity for (q, omega, tau) trajectories.

    Inputs:
      x: (B, T, 10) state trajectory (raw, NOT canonicalized)
      t: (B,) FM time scalar
      cond: (B, cond_dim) conditioning vector

    Output: (B, T, 9) velocity:
      channels 0:3 = body-frame attitude angular velocity (so(3) tangent)
      channels 3:6 = omega velocity (Euclidean)
      channels 6:9 = tau velocity (Euclidean)
    """
    def __init__(self, seq_len, cond_dim, hidden=128, n_layers=4, n_heads=4):
        super().__init__()
        # input projection: state is 10-dim (quat + omega + tau)
        self.in_proj = nn.Linear(10, hidden)
        self.pos_emb = nn.Parameter(torch.randn(1, seq_len, hidden) * 0.02)
        self.t_embed = nn.Sequential(nn.Linear(1, hidden), nn.SiLU(),
                                      nn.Linear(hidden, hidden))
        self.cond_embed = nn.Sequential(nn.Linear(cond_dim, hidden), nn.SiLU(),
                                         nn.Linear(hidden, hidden))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=n_heads, dim_feedforward=hidden * 4,
            batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, n_layers)
        self.out_proj = nn.Linear(hidden, 9)        # 3 + 3 + 3

    def forward(self, x, t, cond):
        # x: (B, T, 10);  t: (B,);  cond: (B, cond_dim)
        h = self.in_proj(x) + self.pos_emb[:, :x.size(1), :]
        emb = self.t_embed(t.unsqueeze(-1)) + self.cond_embed(cond)
        h = h + emb[:, None, :]
        h = self.transformer(h)
        return self.out_proj(h)                     # (B, T, 9)


# -------------------------------------------------------------------------
# normalization helpers (only omega/tau need normalization; quat is unit-norm)
# -------------------------------------------------------------------------

def compute_omega_tau_stats(traj):
    """traj: (N, 10, T) or (N, T, 10) — handle both. Returns mean/std for
    omega/tau channels."""
    if traj.shape[-1] == 10:                       # (N, T, 10)
        ot = traj[..., 4:10]
        m = ot.mean(dim=(0, 1), keepdim=True)
        s = ot.std(dim=(0, 1), keepdim=True).clamp_min(1e-6)
    else:                                          # (N, 10, T)
        ot = traj[:, 4:10, :]
        m = ot.mean(dim=(0, 2), keepdim=True)
        s = ot.std(dim=(0, 2), keepdim=True).clamp_min(1e-6)
    return m, s


# -------------------------------------------------------------------------
# Joint Riemannian FM trainer
# -------------------------------------------------------------------------

def train_joint_fm(net, traj_data, cond_data, n_epochs, batch_size=64,
                   lr=1e-3, device="cpu", log_every=50):
    """traj_data: (N, T, 10) raw trajectories (quat normalized, omega/tau raw).
    cond_data:   (N, cond_dim)."""
    traj_data = traj_data.to(device); cond_data = cond_data.to(device)
    N, T, _ = traj_data.shape

    # normalize omega/tau (in-place on a copy)
    ot_mean, ot_std = compute_omega_tau_stats(traj_data)
    traj_n = traj_data.clone()
    traj_n[..., 4:10] = (traj_n[..., 4:10] - ot_mean) / ot_std

    manifold = SO3()
    path = GeodesicProbPath(scheduler=CondOTScheduler(), manifold=manifold)

    net = net.to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

    for ep in range(n_epochs):
        idx = torch.randperm(N, device=device)[:batch_size]
        x_1 = traj_n[idx]                          # (B, T, 10) normalized
        c = cond_data[idx]
        B = x_1.size(0)

        # split channels
        q1 = x_1[..., 0:4]                          # (B, T, 4)
        ot1 = x_1[..., 4:10]                        # (B, T, 6) normalized

        # ---- noise sampling ----
        # SO(3) noise per (B, T)
        q0 = manifold.projx(_so3_exp(torch.randn(B, T, 3, device=device) * 1.0))
        # Euclidean noise for omega/tau
        ot0 = torch.randn_like(ot1)

        # FM time scalar per sample
        t = torch.rand(B, device=device)

        # ---- attitude path via library (vmap-aware, so flatten time axis) ---
        x_0_flat = q0.reshape(B * T, 4)
        x_1_flat = q1.reshape(B * T, 4)
        t_flat = t.unsqueeze(-1).expand(B, T).reshape(B * T)
        sample = path.sample(x_0=x_0_flat, x_1=x_1_flat, t=t_flat)
        q_t = sample.x_t.reshape(B, T, 4)
        dq_t = sample.dx_t.reshape(B, T, 4)
        # body-frame angular velocity = 2 * (q_t^{-1} * dq_t).vector_part
        rate = _quat_mul(_quat_conj(q_t), dq_t)
        v_att_target = 2.0 * rate[..., 1:]          # (B, T, 3)

        # ---- omega/tau linear interpolation ----
        t_b = t.view(B, 1, 1)
        ot_t = (1 - t_b) * ot0 + t_b * ot1          # (B, T, 6)
        v_ot_target = ot1 - ot0                     # (B, T, 6)

        # ---- assemble model input and prediction ----
        state_t = torch.cat([q_t, ot_t], dim=-1)    # (B, T, 10)
        v_pred = net(state_t, t, c)                 # (B, T, 9)
        v_att_pred = v_pred[..., 0:3]
        v_ot_pred = v_pred[..., 3:9]

        loss_att = F.mse_loss(v_att_pred, v_att_target)
        loss_ot = F.mse_loss(v_ot_pred, v_ot_target)
        loss = loss_att + loss_ot

        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sched.step()

        if ep % log_every == 0 or ep == n_epochs - 1:
            print(f"  ep {ep:4d}  loss = {loss.item():.5f}  "
                  f"(att {loss_att.item():.5f}, ot {loss_ot.item():.5f})")

    stats = dict(ot_mean=ot_mean.cpu(), ot_std=ot_std.cpu())
    return net, stats


# -------------------------------------------------------------------------
# Joint sampler with manifold-aware CVF boundary anchoring
# -------------------------------------------------------------------------

@torch.no_grad()
def sample_joint(net, cond, stats, seq_len, n_steps=50, device="cpu",
                 cvf_initial_attitude=True, cvf_zero_rates=True,
                 q0_from_cond_idx=(0, 4), cvf_final_attitude=True,
                 qT_from_cond_idx=(4, 8)):
    """Sample full trajectories. cond: (B, cond_dim) raw conditioning.
    q0_from_cond_idx / qT_from_cond_idx: slices of cond holding the initial
      and goal attitudes used by the CVF boundary anchors."""
    net.eval()
    cond = cond.to(device); B = cond.size(0)
    manifold = SO3()
    ot_mean = stats["ot_mean"].to(device)
    ot_std = stats["ot_std"].to(device)

    # initialize noise
    q = manifold.projx(_so3_exp(torch.randn(B, seq_len, 3, device=device) * 1.0))
    ot = torch.randn(B, seq_len, 6, device=device)

    dt = 1.0 / n_steps
    EPS = 1e-3
    for i in range(n_steps):
        t_val = i * dt
        t = torch.full((B,), t_val, device=device)

        state = torch.cat([q, ot], dim=-1)          # (B, T, 10)
        v = net(state, t, cond)                     # (B, T, 9)
        v_att = v[..., 0:3].clone()
        v_ot = v[..., 3:9].clone()

        # ---- CVF: boundary attitude anchors (geodesic velocity, Eq. 21) ----
        if cvf_initial_attitude:
            q0 = cond[:, q0_from_cond_idx[0]:q0_from_cond_idx[1]]
            denom = max(1.0 - t_val, EPS)
            v_att[:, 0, :] = manifold.logmap(q[:, 0, :], q0) / denom
        if cvf_final_attitude:
            qT = cond[:, qT_from_cond_idx[0]:qT_from_cond_idx[1]]
            denom = max(1.0 - t_val, EPS)
            v_att[:, -1, :] = manifold.logmap(q[:, -1, :], qT) / denom

        # ---- CVF: zero rates at endpoints (in NORMALIZED space) ----
        if cvf_zero_rates:
            # zero omega in physical space => normalized value = -mean/std for omega
            zero_om_norm = (-ot_mean[..., 0:3] / ot_std[..., 0:3]).squeeze()
            zero_om_norm = zero_om_norm.expand(B, 3)
            denom = max(1.0 - t_val, EPS)
            v_ot[:, 0, 0:3] = (zero_om_norm - ot[:, 0, 0:3]) / denom
            v_ot[:, -1, 0:3] = (zero_om_norm - ot[:, -1, 0:3]) / denom

        # advance
        q = manifold.projx(_quat_mul(q, _so3_exp(dt * v_att)))
        ot = ot + dt * v_ot

    # denormalize omega/tau
    ot_phys = ot * ot_std + ot_mean

    state_out = torch.cat([q, ot_phys], dim=-1)     # (B, T, 10)
    return state_out


def sample_joint_trace(net, cond, stats, seq_len, n_steps=50, device="cpu",
                       cvf_initial_attitude=True, cvf_zero_rates=True,
                       q0_from_cond_idx=(0, 4), cvf_final_attitude=True,
                       qT_from_cond_idx=(4, 8)):
    """Like sample_joint but returns a LIST of (T,10) numpy states, one per flow
    step -- for animating the generation process (noise -> route). Batch elem 0."""
    net.eval()
    cond = cond.to(device); B = cond.size(0)
    manifold = SO3()
    ot_mean = stats["ot_mean"].to(device); ot_std = stats["ot_std"].to(device)
    q = manifold.projx(_so3_exp(torch.randn(B, seq_len, 3, device=device) * 1.0))
    ot = torch.randn(B, seq_len, 6, device=device)
    dt = 1.0 / n_steps; EPS = 1e-3; trace = []
    for i in range(n_steps):
        t_val = i * dt; t = torch.full((B,), t_val, device=device)
        state = torch.cat([q, ot], dim=-1)
        v = net(state, t, cond); v_att = v[..., 0:3].clone(); v_ot = v[..., 3:9].clone()
        if cvf_initial_attitude:
            q0 = cond[:, q0_from_cond_idx[0]:q0_from_cond_idx[1]]
            v_att[:, 0, :] = manifold.logmap(q[:, 0, :], q0) / max(1.0 - t_val, EPS)
        if cvf_final_attitude:
            qT = cond[:, qT_from_cond_idx[0]:qT_from_cond_idx[1]]
            v_att[:, -1, :] = manifold.logmap(q[:, -1, :], qT) / max(1.0 - t_val, EPS)
        if cvf_zero_rates:
            zero_om_norm = (-ot_mean[..., 0:3] / ot_std[..., 0:3]).squeeze().expand(B, 3)
            denom = max(1.0 - t_val, EPS)
            v_ot[:, 0, 0:3] = (zero_om_norm - ot[:, 0, 0:3]) / denom
            v_ot[:, -1, 0:3] = (zero_om_norm - ot[:, -1, 0:3]) / denom
        q = manifold.projx(_quat_mul(q, _so3_exp(dt * v_att))); ot = ot + dt * v_ot
        snap = torch.cat([q, ot * ot_std + ot_mean], dim=-1)[0].detach().cpu().numpy()
        trace.append(snap)                              # (T,10)
    return trace


# -------------------------------------------------------------------------
# Smoke test
# -------------------------------------------------------------------------

def _smoke_test():
    """Synthetic joint SO(3)xR^6 dataset: 100 trajectories, each is SLERP
    from identity to a target attitude with smooth angular velocity profile.
    """
    torch.manual_seed(0)
    T = 10
    cond_dim = 4              # condition = target quaternion only

    n = 100
    # random target attitudes
    omegas = torch.randn(n, 3) * 1.0
    targets = _so3_exp(omegas)

    # ground truth: SLERP from identity + linearly-decreasing omega proxy + zero tau
    ts = torch.linspace(0, 1, T)
    q_traj = torch.zeros(n, T, 4)
    om_traj = torch.zeros(n, T, 3)
    tau_traj = torch.zeros(n, T, 3)
    q_id = torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(n, 4)
    for k in range(T):
        rel = SO3().logmap(q_id, targets)
        q_traj[:, k] = SO3().expmap(q_id, ts[k] * rel)
        # angular velocity along the geodesic ~ constant body-rate
        om_traj[:, k] = rel / T               # crude smooth profile
    om_traj[:, 0] = 0; om_traj[:, -1] = 0
    traj = torch.cat([q_traj, om_traj, tau_traj], dim=-1)      # (n, T, 10)
    cond = targets                                              # (n, 4)

    print(f"smoke dataset: traj {tuple(traj.shape)}, cond {tuple(cond.shape)}")
    net = JointVelocityNet(seq_len=T, cond_dim=cond_dim, hidden=64, n_layers=2)
    net, stats = train_joint_fm(net, traj, cond, n_epochs=400,
                                 batch_size=32, lr=1e-3, log_every=80)

    # sample
    test_targets = _so3_exp(torch.randn(16, 3) * 1.0)
    samples = sample_joint(net, test_targets, stats, seq_len=T, n_steps=50,
                            cvf_initial_attitude=False,         # no initial-q in cond
                            cvf_final_attitude=False,           # measure the learned endpoint
                            cvf_zero_rates=True)
    # final attitude error
    final_q = samples[:, -1, 0:4]
    err = SO3().logmap(final_q, test_targets).norm(dim=-1)
    deg = torch.rad2deg(err)
    print(f"\nfinal attitude err (no init-CVF): mean {deg.mean():.2f}°, "
          f"median {deg.median():.2f}°, max {deg.max():.2f}°")

    # check omega channels: should be small at endpoints (rate anchoring)
    om_init = samples[:, 0, 4:7].norm(dim=-1)
    om_final = samples[:, -1, 4:7].norm(dim=-1)
    print(f"||omega(0)|| mean: {om_init.mean():.3f},  ||omega(T)|| mean: {om_final.mean():.3f}")
    if om_init.mean() < 0.5 and om_final.mean() < 0.5:
        print("PASS — rate anchoring is working")
    else:
        print("PARTIAL — rate anchoring needs longer training or wider net")


if __name__ == "__main__":
    _smoke_test()
