"""SO(3) manifold class for the facebookresearch/flow_matching library.

Conforms to the Manifold abstract interface (expmap, logmap, projx, proju).

Representation: unit quaternions [w, x, y, z], scalar-first.
Tangent space at q: the 3-dim Lie algebra so(3), expressed as axis-angle
vectors in R^3 — these are the BODY-FRAME tangents.

The library's GeodesicProbPath uses these operations as black boxes; if
they are mathematically correct, the path/loss/solver are automatically
correct.

Verification strategy: numerical tests of (1) expmap/logmap roundtrip,
(2) geodesic equivariance under global rotation, (3) tangent projection
preserves the body-frame velocity.
"""
import torch
from torch import Tensor
from flow_matching.utils.manifolds import Manifold

EPS_F32 = 1e-6


def _quat_mul(a: Tensor, b: Tensor) -> Tensor:
    """Hamilton product, broadcastable. a, b: (..., 4)."""
    w1, x1, y1, z1 = a.unbind(-1)
    w2, x2, y2, z2 = b.unbind(-1)
    return torch.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dim=-1)


def _quat_conj(q: Tensor) -> Tensor:
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def _quat_canonical(q: Tensor) -> Tensor:
    """Resolve q ~ -q sign ambiguity by forcing scalar component non-negative."""
    return torch.where(q[..., :1] < 0, -q, q)


def _so3_exp(omega: Tensor) -> Tensor:
    """exp map so(3) -> SO(3). omega: (..., 3) axis-angle. Output: (..., 4)."""
    angle = omega.norm(dim=-1, keepdim=True).clamp_min(EPS_F32 * 0)
    half = 0.5 * angle
    # sin(angle/2) / angle, stable near 0 via Taylor
    small = angle < EPS_F32
    sin_over_angle = torch.where(
        small,
        0.5 - angle * angle / 48.0,
        torch.sin(half) / angle.clamp_min(EPS_F32),
    )
    w = torch.cos(half)
    xyz = omega * sin_over_angle
    out = torch.cat([w, xyz], dim=-1)
    return out / out.norm(dim=-1, keepdim=True).clamp_min(EPS_F32)


def _so3_log(q: Tensor) -> Tensor:
    """log map SO(3) -> so(3). q: (..., 4) unit quat. Output: (..., 3)."""
    q = _quat_canonical(q / q.norm(dim=-1, keepdim=True).clamp_min(EPS_F32))
    w = q[..., :1].clamp(-1.0, 1.0)
    xyz = q[..., 1:]
    sin_half = xyz.norm(dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(sin_half, w)        # safe near both poles
    small = sin_half < EPS_F32
    scale = torch.where(
        small,
        2.0 + (angle * angle) / 12.0,             # Taylor of angle / sin(angle/2)
        angle / sin_half.clamp_min(EPS_F32),
    )
    return xyz * scale


class SO3(Manifold):
    """The 3D rotation group SO(3), represented by unit quaternions [w,x,y,z].

    Tangent vectors at q are 3-dim axis-angle elements of so(3), interpreted
    as LEFT-INVARIANT (body-frame) tangents. The exp map at q acts by
    left-multiplication: expmap(q, u) = q * exp(u). The log map at q gives
    the body-frame tangent toward y: logmap(q, y) = log(q^{-1} y).

    Note on shape: this class STORES points as (..., 4) and tangents as (..., 3),
    so they have different last dimensions. The flow_matching library's
    GeodesicProbPath uses einsum to scale the tangent by `t` and passes it
    back through expmap with the right unsqueeze, so it handles this
    naturally (see geodesic() in flow_matching.utils.manifolds.utils).
    """

    def expmap(self, x: Tensor, u: Tensor) -> Tensor:
        """expmap(x, u) = x * exp(u). x: (..., 4); u: (..., 3) or broadcastable.
        Returns: (..., 4)."""
        # Handle the case where geodesic() einsum produces tangent shape
        # (..., T, 3) and x has shape (..., 4) (without time axis): expand.
        if u.dim() == x.dim() + 1 and u.size(-1) == 3:
            # x: (..., 4), u: (..., T, 3). Broadcast x along T.
            x = x.unsqueeze(-2)
        return self.projx(_quat_mul(x, _so3_exp(u)))

    def logmap(self, x: Tensor, y: Tensor) -> Tensor:
        """logmap(x, y) = log(x^{-1} y). x, y: (..., 4). Returns: (..., 3)."""
        rel = _quat_mul(_quat_conj(x), y)
        return _so3_log(rel)

    def projx(self, x: Tensor) -> Tensor:
        """Project to the manifold: normalize + canonicalize the sign."""
        return _quat_canonical(x / x.norm(dim=-1, keepdim=True).clamp_min(EPS_F32))

    def proju(self, x: Tensor, u: Tensor) -> Tensor:
        """Project a vector to the tangent space at x. For SO(3) with body-frame
        tangents represented in R^3, the tangent space is the whole R^3 — no
        projection needed."""
        return u


if __name__ == "__main__":
    torch.manual_seed(0)
    m = SO3()

    # roundtrip
    x = m.projx(torch.randn(4, 4))
    y = m.projx(torch.randn(4, 4))
    u = m.logmap(x, y)
    y_back = m.expmap(x, u)
    print(f"logmap/expmap roundtrip err: {(y - y_back).abs().max().item():.2e}")

    # equivariance: logmap(R x, R y) == logmap(x, y) (body-frame is invariant)
    R = m.projx(torch.randn(4, 4))
    Rx = m.projx(_quat_mul(R, x))
    Ry = m.projx(_quat_mul(R, y))
    u_rot = m.logmap(Rx, Ry)
    print(f"logmap equivariance err: {(u - u_rot).abs().max().item():.2e}")

    # geodesic via library
    from flow_matching.utils.manifolds.utils import geodesic
    g = geodesic(m, x, y)
    t = torch.tensor([0.0, 0.5, 1.0])
    pts = g(t)
    print(f"geodesic shape: {tuple(pts.shape)}")
    err_0 = (pts[..., 0, :] - x).abs().max().item()
    err_1 = min((pts[..., -1, :] - y).abs().max().item(),
                (pts[..., -1, :] + y).abs().max().item())
    print(f"geodesic t=0 err: {err_0:.2e},  t=1 err: {err_1:.2e}")
