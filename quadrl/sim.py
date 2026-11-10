"""Simulación en lazo cerrado con la misma temporización que Simulink.

Integra con RK4 a dt_sim, aplica el controlador a dt_ctrl con retención de
orden cero y modela el retardo del lazo como un búfer de comandos.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from .dynamics import N_U, QuadParams, rk4_step


def n_substeps(dt_ctrl: float, dt_sim: float) -> int:
    n = round(dt_ctrl / dt_sim)
    if not np.isclose(n * dt_sim, dt_ctrl):
        raise ValueError("dt_ctrl debe ser múltiplo entero de dt_sim")
    return n


class DelayLine:
    """Retardo de d pasos de control; arranca lleno con el comando de hover."""

    def __init__(self, steps: int, u_init: np.ndarray):
        self.buf = deque([u_init.copy() for _ in range(steps)], maxlen=steps + 1)
        self.steps = steps

    def __call__(self, u):
        if self.steps == 0:
            return u
        self.buf.append(u.copy())
        return self.buf.popleft()


def simulate(controller, P: QuadParams, x0, ref, t_final, dt_sim, dt_ctrl, loop_delay=0.0, f_ext=None):
    nsub = n_substeps(dt_ctrl, dt_sim)
    n_ctrl = int(round(t_final / dt_ctrl))
    delay = DelayLine(int(round(loop_delay / dt_ctrl)), np.full(N_U, P.hover_thrust))
    controller.reset()

    x = np.asarray(x0, float).copy()
    T, X, U = [0.0], [x.copy()], []
    for k in range(n_ctrl):
        u = delay(controller(x, ref))
        for _ in range(nsub):
            x = rk4_step(x, u, dt_sim, P, f_ext)
        U.append(u)
        T.append((k + 1) * dt_ctrl)
        X.append(x.copy())
    U.append(U[-1])
    return np.array(T), np.array(X), np.array(U)
