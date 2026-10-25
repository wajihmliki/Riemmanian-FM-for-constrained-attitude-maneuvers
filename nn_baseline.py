"""nn_baseline.py -- nearest-neighbor retrieval warm-start (no training).

Given a test instance, find the training instance with the closest conditioning
vector (start, target, cone geometry) in Euclidean distance and use its stored
trajectory directly as the SCvx warm-start. This tests whether performance comes
from learning or from retrieving similar solved examples.

make_nn_init(train_pkl) -> fn(inst) -> xi (3, T), the same interface as
eval_warmstart.load_fm.
"""
import pickle
import numpy as np
from preflight_core import qnorm, qmul, qconj, log_so3

_PAD_DIR = np.array([0., 0., 1.])


def _cond_of(inst, m_max):
    cones = list(inst['cones'])
    if len(cones) < m_max:
        cones = cones + [(_PAD_DIR.copy(), 1.0)] * (m_max - len(cones))
    elif len(cones) > m_max:
        cones = cones[:m_max]
    dirs = np.concatenate([c[0] for c in cones]); cos = np.array([c[1] for c in cones])
    return np.concatenate([inst['q0'], inst['q_tgt'], dirs, cos]).astype(np.float32)


def make_nn_init(train_pkl, reanchor=True):
    """Returns fn(inst) -> xi (3, T) from the nearest training trajectory by conditioning."""
    d = pickle.load(open(train_pkl, 'rb'))
    conds = np.asarray(d['conditions'], dtype=np.float32)          # (Ntr, cond_dim)
    trajs = np.asarray(d['trajectories'], dtype=np.float32)        # (Ntr, 10, T) or (Ntr, T, 10)
    if trajs.shape[1] != 10 and trajs.shape[2] == 10:             # -> (Ntr, 10, T)
        trajs = trajs.transpose(0, 2, 1)
    m_max = (conds.shape[1] - 8) // 4

    def fn(inst):
        c = _cond_of(inst, m_max)
        j = int(np.argmin(np.linalg.norm(conds - c[None, :], axis=1)))   # nearest by conditioning
        q = trajs[j, 0:4, :]                                            # (4, T)
        T = q.shape[1]
        qs = [qnorm(q[:, k]) for k in range(T)]
        for k in range(1, T):                                          # sign-continuity
            if np.dot(qs[k], qs[k - 1]) < 0: qs[k] = -qs[k]
        base = qs[0] if reanchor else inst['q0']
        xi = np.stack([log_so3(qmul(qconj(base), qs[k])) for k in range(T)], axis=1)
        return xi
    return fn
