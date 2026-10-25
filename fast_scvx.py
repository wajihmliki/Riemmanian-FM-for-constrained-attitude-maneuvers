"""fast_scvx.py -- compile-once SCvx.

Builds the convex subproblem ONCE with cvxpy Parameters (DPP), then reuses it
across every SCvx iteration and every problem instance of the same (N, ncone);
only parameter values change. Same full rigid-body dynamics (gyroscopic term),
same soft keep-out formulation as preflight_core.scvx_solve.

Usage:
    S = FastSCvx(N=60, ncone=5, dt=0.6, tau_max=0.15)
    res = S.solve(inst, xi_init=None)        # inst as in generate_corridor
    # feasibility is decided by preflight_core.verify(inst, res['tau'])
"""
import numpy as np
import cvxpy as cp
from preflight_core import (exp_so3, log_so3, qmul, qconj, qnorm, rotmat,
                            _step_omega, _step_xi, _num_jac)


def _skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


class FastSCvx:
    def __init__(self, N, ncone, dt=0.6, tau_max=0.12, J=None,
                 keepout_penalty=1e3, solver='CLARABEL'):
        self.N = N; self.ncone = ncone; self.dt = dt; self.tau_max = tau_max
        self.J = np.diag([10., 12., 15.]) if J is None else J
        self.Jinv = np.linalg.inv(self.J)
        self.solver = solver
        self._build(keepout_penalty)

    def _build(self, penalty):
        N = self.N; nc = self.ncone
        xi = cp.Variable((3, N + 1)); om = cp.Variable((3, N + 1)); tau = cp.Variable((3, N))
        dxi = cp.Variable((3, N + 1)); dom = cp.Variable((3, N + 1))   # trust deltas (DPP-safe)
        S = cp.Variable((nc, N), nonneg=True)                          # keep-out slacks
        self.xi, self.om, self.tau, self.S = xi, om, tau, S

        # parameters
        self.Aom = [cp.Parameter((3, 3)) for _ in range(N)]
        self.Bm  = [cp.Parameter((3, 3)) for _ in range(N)]
        self.com = [cp.Parameter(3) for _ in range(N)]
        self.Gxi = [cp.Parameter((3, 3)) for _ in range(N)]
        self.Gom = [cp.Parameter((3, 3)) for _ in range(N)]
        self.cxi = [cp.Parameter(3) for _ in range(N)]
        self.kg  = [[cp.Parameter(3) for _ in range(N)] for _ in range(nc)]   # node k=1..N -> idx k-1
        self.kr  = [[cp.Parameter() for _ in range(N)] for _ in range(nc)]
        self.xinom = cp.Parameter((3, N + 1)); self.omnom = cp.Parameter((3, N + 1))
        self.xitgt = cp.Parameter(3); self.w0 = cp.Parameter(3)
        self.tw = cp.Parameter(nonneg=True)

        cons = [xi[:, 0] == 0, om[:, 0] == self.w0, xi[:, N] == self.xitgt, om[:, N] == 0,
                dxi == xi - self.xinom, dom == om - self.omnom]
        for k in range(N):
            cons.append(om[:, k+1] == self.Aom[k] @ om[:, k] + self.Bm[k] @ tau[:, k] + self.com[k])
            cons.append(xi[:, k+1] == self.Gxi[k] @ xi[:, k] + self.Gom[k] @ om[:, k] + self.cxi[k])
            cons.append(cp.norm(tau[:, k], 2) <= self.tau_max)
        for ci in range(nc):
            for k in range(1, N + 1):
                cons.append(self.kg[ci][k-1] @ xi[:, k] - S[ci, k-1] <= self.kr[ci][k-1])

        obj = cp.Minimize(cp.sum([cp.sum_squares(tau[:, k]) for k in range(N)]) * self.dt
                          + self.tw * (cp.sum_squares(dxi) + cp.sum_squares(dom))
                          + penalty * cp.sum(S))
        self.prob = cp.Problem(obj, cons)
        assert self.prob.is_dcp(dpp=True), "problem is not DPP"

    def _set_params(self, inst, xi_nom, om_nom, tau_nom, keepout_margin=0.30):
        N = self.N; dt = self.dt; J = self.J; Jinv = self.Jinv
        q0 = inst['q0']; b_body = inst['b_body']; cones = inst['cones']
        self.xitgt.value = inst['xi_target']; self.w0.value = inst['w0']
        self.xinom.value = xi_nom; self.omnom.value = om_nom
        for k in range(N):
            w = om_nom[:, k]
            Aom = np.eye(3) - dt * Jinv @ (_skew(w) @ J - _skew(J @ w))   # analytic gyroscopic jac
            Bm = dt * Jinv
            f0 = _step_omega(w, tau_nom[:, k], dt, J, Jinv)
            self.Aom[k].value = Aom; self.Bm[k].value = Bm
            self.com[k].value = f0 - Aom @ w - Bm @ tau_nom[:, k]
            g0 = _step_xi(xi_nom[:, k], w, dt)
            Gxi = _num_jac(lambda x: _step_xi(x, w, dt), xi_nom[:, k])
            Gom = _num_jac(lambda x: _step_xi(xi_nom[:, k], x, dt), w)
            self.Gxi[k].value = Gxi; self.Gom[k].value = Gom
            self.cxi[k].value = g0 - Gxi @ xi_nom[:, k] - Gom @ w
        for ci, (cdir, cos_exc) in enumerate(cones):
            for k in range(1, N + 1):
                qk = qnorm(qmul(q0, exp_so3(xi_nom[:, k])))
                c0 = (rotmat(qk) @ b_body) @ cdir
                if c0 > cos_exc - keepout_margin:
                    grad = np.zeros(3)
                    for j in range(3):
                        e = np.zeros(3); e[j] = 1e-5
                        qp = qnorm(qmul(q0, exp_so3(xi_nom[:, k] + e)))
                        grad[j] = ((rotmat(qp) @ b_body) @ cdir - c0) / 1e-5
                    self.kg[ci][k-1].value = grad
                    self.kr[ci][k-1].value = float(cos_exc - c0 + grad @ xi_nom[:, k])
                else:
                    self.kg[ci][k-1].value = np.zeros(3)
                    self.kr[ci][k-1].value = 1e3        # inactive

    def solve(self, inst, xi_init=None, n_iter=20, trust_weight=5.0, tol=1e-3, max_iters=200):
        N = self.N
        xit = inst['xi_target']
        xi_nom = xi_init.copy() if xi_init is not None else np.outer(xit, np.linspace(0, 1, N + 1))
        om_nom = np.zeros((3, N + 1)); om_nom[:, :-1] = (xi_nom[:, 1:] - xi_nom[:, :-1]) / self.dt
        tau_nom = np.zeros((3, N))
        self.tw.value = trust_weight
        costs = []; last = None; status = 'no_iter'

        # Configure solver kwargs dynamically
        solver_opts = {'verbose': False}
        solver_name = self.solver.upper() if isinstance(self.solver, str) else str(self.solver)

        if solver_name == 'ECOS':
            solver_opts['max_iters'] = max_iters
            solver_opts['abstol'] = 1e-7
            solver_opts['reltol'] = 1e-6
        elif solver_name == 'CLARABEL':
            solver_opts['max_iter'] = max_iters
            solver_opts['warm_start'] = True
        else:
            solver_opts['warm_start'] = True

        for it in range(n_iter):
            self._set_params(inst, xi_nom, om_nom, tau_nom)
            try:
                self.prob.solve(solver=getattr(cp, self.solver) if isinstance(self.solver, str) else self.solver, **solver_opts)
            except cp.SolverError:
                status = 'solver_error'; break
            except Exception:
                status = 'solver_exception'; break

            if self.prob.status not in ('optimal', 'optimal_inaccurate'):
                status = self.prob.status; break

            status = self.prob.status
            ctrl = float(np.sum(np.linalg.norm(self.tau.value, axis=0)**2) * self.dt)
            costs.append(ctrl)
            last = (self.xi.value.copy(), self.om.value.copy(), self.tau.value.copy())
            xi_nom, om_nom, tau_nom = self.xi.value, self.om.value, self.tau.value
            if len(costs) >= 2 and abs(costs[-1]-costs[-2]) / max(abs(costs[-2]), 1e-9) < tol:
                break

        if last is None:
            return dict(xi=None, om=None, tau=None, costs=costs, status=status, iters=len(costs))
        return dict(xi=last[0], om=last[1], tau=last[2], costs=costs, status=status, iters=len(costs))
