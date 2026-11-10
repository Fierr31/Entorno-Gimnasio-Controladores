"""Linealización, discretización y controladores LQR / LQI de referencia.

Estos controladores son la línea base contra la que se comparará la política
de RL. Deben reproducir las ganancias de tu diseño en MATLAB: compara
`K` con tu `dlqr`/`lqi` antes de confiar en cualquier figura.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm, solve_discrete_are

from .dynamics import N_U, N_X, QuadParams, f, hover_state

OUT_IDX = [0, 1, 2, 8]   # salidas integradas por el LQI: x, y, z, psi


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def linearize(P: QuadParams, eps: float = 1e-6):
    """Jacobianos numéricos (diferencias centrales) en hover."""
    x0, u0 = hover_state(P), np.full(N_U, P.hover_thrust)
    A, B = np.zeros((N_X, N_X)), np.zeros((N_X, N_U))
    for i in range(N_X):
        d = np.zeros(N_X); d[i] = eps
        A[:, i] = (f(x0 + d, u0, P) - f(x0 - d, u0, P)) / (2 * eps)
    for j in range(N_U):
        d = np.zeros(N_U); d[j] = eps
        B[:, j] = (f(x0, u0 + d, P) - f(x0, u0 - d, P)) / (2 * eps)
    return A, B


def c2d(A, B, dt):
    """Discretización ZOH exacta (equivalente a c2d(..., 'zoh') en MATLAB)."""
    n, m = B.shape
    M = np.zeros((n + m, n + m))
    M[:n, :n], M[:n, n:] = A, B
    E = expm(M * dt)
    return E[:n, :n], E[:n, n:]


def dlqr(A, B, Q, R):
    S = solve_discrete_are(A, B, Q, R)
    K = np.linalg.solve(R + B.T @ S @ B, B.T @ S @ A)
    return K, S


def reference_state(P: QuadParams, ref) -> np.ndarray:
    """Estado de equilibrio para r = [x, y, z, psi]."""
    x = hover_state(P, pos=ref[:3])
    x[8] = ref[3]
    return x


def state_error(x, x_ref):
    e = x - x_ref
    e[8] = wrap(e[8])
    return e


class LQR:
    name = "LQR"

    def __init__(self, P: QuadParams, cfg: dict, dt: float):
        self.P, self.dt = P, dt
        A, B = linearize(P)
        self.Ad, self.Bd = c2d(A, B, dt)
        self.Q, self.R = np.diag(cfg["lqr"]["Q_diag"]), np.diag(cfg["lqr"]["R_diag"])
        self.K, _ = dlqr(self.Ad, self.Bd, self.Q, self.R)
        self.u_hover = np.full(N_U, P.hover_thrust)

    def reset(self):
        pass

    def __call__(self, x, ref):
        u = self.u_hover - self.K @ state_error(x, reference_state(self.P, ref))
        return np.clip(u, self.P.thrust_min, self.P.thrust_max)


class LQI(LQR):
    """LQR aumentado con integradores discretos sobre [x, y, z, psi].

    Anti-windup condicional: el integrador se congela en los pasos donde
    algún motor queda saturado. AJUSTA a tu esquema si es distinto.
    """
    name = "LQI"

    def __init__(self, P, cfg, dt):
        super().__init__(P, cfg, dt)
        C = np.zeros((len(OUT_IDX), N_X)); C[range(len(OUT_IDX)), OUT_IDX] = 1.0
        ni = C.shape[0]
        Aa = np.block([[self.Ad, np.zeros((N_X, ni))], [-dt * C, np.eye(ni)]])
        Ba = np.vstack([self.Bd, np.zeros((ni, N_U))])
        Qa = np.block([[self.Q, np.zeros((N_X, ni))],
                       [np.zeros((ni, N_X)), np.diag(cfg["lqi"]["Qi_diag"])]])
        Ka, _ = dlqr(Aa, Ba, Qa, self.R)
        self.Kx, self.Ki = Ka[:, :N_X], Ka[:, N_X:]
        self.Aa, self.Ba = Aa, Ba
        self.reset()

    def reset(self):
        self.xi = np.zeros(len(OUT_IDX))

    def __call__(self, x, ref):
        e = state_error(x, reference_state(self.P, ref))
        u_raw = self.u_hover - self.Kx @ e - self.Ki @ self.xi
        u = np.clip(u_raw, self.P.thrust_min, self.P.thrust_max)
        if np.allclose(u, u_raw):
            y_err = -e[OUT_IDX]                 # r - y (psi ya envuelto)
            self.xi = self.xi + self.dt * y_err
        return u
