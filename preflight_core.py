"""SCvx solver and nonlinear feasibility certificate for constrained attitude slews.

  * The SCvx solver uses the full rigid-body dynamics, including the gyroscopic
    term (omega x J omega), re-linearized around the nominal every iteration
    (numerical Jacobians).
  * Feasibility is never judged from solver status or the linearized state. It is
    judged only by forward-integrating the commanded torque under the full
    nonlinear dynamics (RK4, fine substeps) and checking the result (verify()).
  * Every check is applied identically to every initialization method.

Coordinates: the solver optimizes in the body-frame algebra xi = log(q0^{-1} q_k)
relative to the fixed start q0, with body rates omega and body torque tau.
Keep-out is evaluated on the exact boresight R(q_k) b_body.
"""
import numpy as np

# Terminal-attitude gate of the feasibility certificate (deg). The paper uses
# 2 deg throughout and 5 deg for the robustness study; evaluation scripts expose
# it as --tol_att.
TOL_ATT_DEG = 2.0

# ----------------------------------------------------------------------
# SO(3) helpers (quaternion = [w,x,y,z], body->inertial)
# ----------------------------------------------------------------------
def exp_so3(rv):
    """rotation vector -> unit quaternion."""
    a = np.linalg.norm(rv)
    if a < 1e-9:
        q = np.array([1.0, rv[0]/2, rv[1]/2, rv[2]/2])
        return q / np.linalg.norm(q)
    ax = rv / a
    return np.array([np.cos(a/2), *(np.sin(a/2)*ax)])

def log_so3(q):
    """unit quaternion -> rotation vector."""
    if q[0] < 0:
        q = -q
    w = np.clip(q[0], -1.0, 1.0)
    s = np.sqrt(max(1 - w*w, 0.0))
    if s < 1e-9:
        return np.zeros(3)
    return 2*np.arccos(w) * q[1:] / s

def qmul(a, b):
    w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
    return np.array([w1*w2 - x1*x2 - y1*y2 - z1*z2,
                     w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2,
                     w1*z2 + x1*y2 - y1*x2 + z1*w2])

def qconj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])

def qnorm(q):
    return q / (np.linalg.norm(q) + 1e-12)

def rotmat(q):
    w, x, y, z = qnorm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
                     [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)]])


# ----------------------------------------------------------------------
# TRUE nonlinear rigid-body dynamics + integrator (the VERIFICATION gate)
# ----------------------------------------------------------------------
def _deriv(q, w, tau, J, Jinv):
    """Continuous-time derivative of (q, w) under full nonlinear dynamics."""
    wdot = Jinv @ (tau - np.cross(w, J @ w))          # Euler's equation
    qdot = 0.5 * qmul(q, np.array([0.0, *w]))         # body-rate kinematics
    return qdot, wdot

def integrate_true(q0, w0, tau_seq, dt, J, substeps=10):
    """Forward-integrate the TRUE dynamics with zero-order-hold torque.

    tau_seq: (3, N) commanded torques. Returns q_traj (4, N+1), w_traj (3, N+1).
    RK4 with `substeps` per control interval.
    """
    N = tau_seq.shape[1]
    Jinv = np.linalg.inv(J)
    q = qnorm(q0.copy()); w = w0.copy()
    qs = [q.copy()]; ws = [w.copy()]
    h = dt / substeps
    for k in range(N):
        tau = tau_seq[:, k]
        for _ in range(substeps):
            kq1, kw1 = _deriv(q, w, tau, J, Jinv)
            kq2, kw2 = _deriv(qnorm(q + 0.5*h*kq1), w + 0.5*h*kw1, tau, J, Jinv)
            kq3, kw3 = _deriv(qnorm(q + 0.5*h*kq2), w + 0.5*h*kw2, tau, J, Jinv)
            kq4, kw4 = _deriv(qnorm(q + h*kq3),     w + h*kw3,     tau, J, Jinv)
            q = qnorm(q + (h/6)*(kq1 + 2*kq2 + 2*kq3 + kq4))
            w = w + (h/6)*(kw1 + 2*kw2 + 2*kw3 + kw4)
        qs.append(q.copy()); ws.append(w.copy())
    return np.array(qs).T, np.array(ws).T


# ----------------------------------------------------------------------
# Discrete one-step maps used by the SOLVER (linearized each iteration)
# ----------------------------------------------------------------------
def _step_omega(w, tau, dt, J, Jinv):
    """Discrete Euler step (includes gyroscopic term)."""
    return w + dt * (Jinv @ (tau - np.cross(w, J @ w)))

def _step_xi(xi, w, dt):
    """Discrete kinematic step in the q0-relative algebra:
       q_{k+1} = q0 exp(xi_{k+1}) = q0 exp(xi_k) exp(dt w_k)
       => xi_{k+1} = log( exp(xi_k) exp(dt w_k) )   (exact, nonlinear/BCH)."""
    q_rel_next = qmul(exp_so3(xi), exp_so3(dt * w))
    return log_so3(q_rel_next)

def _num_jac(f, x, eps=1e-6):
    """Numerical Jacobian of vector function f at x (central differences)."""
    n = x.shape[0]; f0 = f(x); m = f0.shape[0]
    Jm = np.zeros((m, n))
    for i in range(n):
        dx = np.zeros(n); dx[i] = eps
        Jm[:, i] = (f(x + dx) - f(x - dx)) / (2*eps)
    return Jm


# ----------------------------------------------------------------------
# Rigid-body SCvx (TRUE dynamics linearized each iteration)
# ----------------------------------------------------------------------
import cvxpy as cp

def scvx_solve(inst, xi_init=None, n_iter=20, trust_weight=5.0,
               keepout_margin=0.30, tol=1e-3, verbose=False,
               soft_keepout=True, keepout_penalty=1e3, solver='ECOS'):
    """SCvx for a constrained slew. Optimizes (xi, om, tau) in q0-relative
    algebra with Euler (gyroscopic) + exact kinematics linearized per iter.

    Returns dict with xi/om/tau (or None), per-iter control costs, status,
    iters, wall time. Feasibility is NOT decided here -- use verify().
    """
    import time
    J = inst['J']; Jinv = np.linalg.inv(J); dt = inst['dt']; N = inst['N']
    b_body = inst['b_body']; tau_max = inst['tau_max']
    w0 = inst['w0']; xi_target = inst['xi_target']; q0 = inst['q0']
    cones = inst['cones']                                  # list of (dir3, cos_exc)

    if xi_init is not None:
        xi_nom = xi_init.copy()
    else:
        xi_nom = np.outer(xi_target, np.linspace(0, 1, N+1))
    # nominal omega/tau consistent-ish: finite-diff of xi for om, zero tau
    om_nom = np.zeros((3, N+1))
    om_nom[:, :-1] = (xi_nom[:, 1:] - xi_nom[:, :-1]) / dt
    tau_nom = np.zeros((3, N))

    costs = []; t0 = time.time(); status = 'no_iter'
    last = None
    for it in range(n_iter):
        xi = cp.Variable((3, N+1)); om = cp.Variable((3, N+1)); tau = cp.Variable((3, N))
        cons = [xi[:, 0] == 0, om[:, 0] == w0, xi[:, N] == xi_target, om[:, N] == 0]
        for k in range(N):
            # --- Euler (gyroscopic) linearization ---
            f0 = _step_omega(om_nom[:, k], tau_nom[:, k], dt, J, Jinv)
            A_om = _num_jac(lambda x: _step_omega(x, tau_nom[:, k], dt, J, Jinv), om_nom[:, k])
            B    = _num_jac(lambda u: _step_omega(om_nom[:, k], u, dt, J, Jinv), tau_nom[:, k])
            cons.append(om[:, k+1] == f0 + A_om @ (om[:, k]-om_nom[:, k]) + B @ (tau[:, k]-tau_nom[:, k]))
            # --- exact kinematic step linearization ---
            g0 = _step_xi(xi_nom[:, k], om_nom[:, k], dt)
            G_xi = _num_jac(lambda x: _step_xi(x, om_nom[:, k], dt), xi_nom[:, k])
            G_om = _num_jac(lambda x: _step_xi(xi_nom[:, k], x, dt), om_nom[:, k])
            cons.append(xi[:, k+1] == g0 + G_xi @ (xi[:, k]-xi_nom[:, k]) + G_om @ (om[:, k]-om_nom[:, k]))
            cons.append(cp.norm(tau[:, k], 2) <= tau_max)
        # --- keep-out cones (linearized around xi_nom) ---
        slack_terms = []
        for (cdir, cos_exc) in cones:
            for k in range(1, N+1):
                qk = qnorm(qmul(q0, exp_so3(xi_nom[:, k])))
                c0 = (rotmat(qk) @ b_body) @ cdir
                if c0 > cos_exc - keepout_margin:
                    grad = np.zeros(3)
                    for j in range(3):
                        e = np.zeros(3); e[j] = 1e-5
                        qp = qnorm(qmul(q0, exp_so3(xi_nom[:, k]+e)))
                        grad[j] = ((rotmat(qp) @ b_body) @ cdir - c0) / 1e-5
                    lhs = c0 + grad @ (xi[:, k]-xi_nom[:, k])
                    if soft_keepout:
                        s = cp.Variable(nonneg=True)             # slack: always feasible
                        cons.append(lhs <= cos_exc + s)
                        slack_terms.append(s)
                    else:
                        cons.append(lhs <= cos_exc)
        penalty = keepout_penalty * cp.sum(slack_terms) if slack_terms else 0
        obj = cp.Minimize(cp.sum([cp.sum_squares(tau[:, k]) for k in range(N)]) * dt
                          + trust_weight * (cp.sum_squares(xi - xi_nom) + cp.sum_squares(om - om_nom))
                          + penalty)
        prob = cp.Problem(obj, cons)
        try:
            prob.solve(solver=getattr(cp, solver), verbose=False)
        except Exception as ex:
            status = 'solver_exception'; break
        if prob.status not in ('optimal', 'optimal_inaccurate'):
            status = prob.status; break
        status = prob.status
        ctrl = float(np.sum(np.linalg.norm(tau.value, axis=0)**2) * dt)
        costs.append(ctrl)
        last = (xi.value.copy(), om.value.copy(), tau.value.copy())
        xi_nom, om_nom, tau_nom = xi.value, om.value, tau.value
        if len(costs) >= 2 and abs(costs[-1]-costs[-2]) / max(abs(costs[-2]), 1e-9) < tol:
            break
    wall = time.time() - t0
    if last is None:
        return dict(xi=None, om=None, tau=None, costs=costs, status=status, iters=len(costs), wall=wall)
    return dict(xi=last[0], om=last[1], tau=last[2], costs=costs,
                status=status, iters=len(costs), wall=wall)


def verify(inst, tau_seq, substeps=15, tol_att=None, tol_vio=0.5):
    """Forward-integrate COMMANDED torque under TRUE nonlinear dynamics and
    decide feasibility. Returns dict (the ONLY place feasibility is decided).
    Gate: terminal attitude < tol_att (deg, default TOL_ATT_DEG), keep-out
    violation < tol_vio (deg), torque within 2% of tau_max."""
    if tol_att is None:
        tol_att = TOL_ATT_DEG
    if tau_seq is None:
        return dict(feasible=False, att_err=180.0, max_vio=90.0, rate_term=99.0, max_tau=99.0)
    J = inst['J']; dt = inst['dt']; b_body = inst['b_body']
    q0 = inst['q0']; w0 = inst['w0']; cones = inst['cones']
    q_tgt = qnorm(qmul(q0, exp_so3(inst['xi_target'])))
    qs, ws = integrate_true(q0, w0, tau_seq, dt, J, substeps=substeps)
    att_err = float(np.degrees(np.linalg.norm(log_so3(qmul(qconj(qs[:, -1]), q_tgt)))))
    rate_term = float(np.linalg.norm(ws[:, -1]))
    max_tau = float(np.max(np.linalg.norm(tau_seq, axis=0)))
    max_vio = 0.0
    for k in range(qs.shape[1]):
        bi = rotmat(qs[:, k]) @ b_body
        for (cdir, cos_exc) in cones:
            ang = np.degrees(np.arccos(np.clip(bi @ cdir, -1, 1)))
            exc = np.degrees(np.arccos(np.clip(cos_exc, -1, 1)))
            max_vio = max(max_vio, exc - ang)
    max_vio = max(0.0, max_vio)
    feasible = (att_err < tol_att) and (max_vio < tol_vio) and (max_tau <= inst['tau_max']*1.02)
    return dict(feasible=bool(feasible), att_err=att_err, max_vio=max_vio,
                rate_term=rate_term, max_tau=max_tau)
