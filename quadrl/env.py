"""Entorno Gymnasium del cuadricóptero de 16 estados.

Acción: 4 valores en [-1, 1] -> empuje comandado por motor (misma interfaz
que el LQI). Observación (16): error de posición, velocidad, actitud (psi
como error envuelto), velocidades angulares y acción previa.
Recompensa: el mismo costo cuadrático del LQR (Q, R), más penalización
por cambios bruscos de acción y un término por choque.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .controllers import reference_state, state_error
from .dynamics import N_U, load_config, params_from_config, rk4_step
from .sim import DelayLine, n_substeps

# Contrato entre la política y quien la ejecute (el firmware, en su repo privado).
# Cambiar cualquier campo = subir INTERFACE["version"]; el integrador del firmware
# debe rechazar políticas con una versión que no conoce.
INTERFACE = {
    "version": 1,
    "obs": ["ex", "ey", "ez", "vx", "vy", "vz", "phi", "theta", "e_psi",
            "p", "q", "r", "a1_prev", "a2_prev", "a3_prev", "a4_prev"],
    "obs_units": "m, m/s, rad (ZYX, z arriba, e_psi envuelto a [-pi, pi)), rad/s, adim.",
    "position_error": "p - p_ref (inercial)",
    "action": "a_i en [-1, 1] -> T_cmd_i = T_min + (a_i + 1)/2 * (T_max - T_min) [N]",
    "motor_order": "config motors.angles_deg / motors.spin",
}


class QuadEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config_path=None, task="hover", randomize=False,
                 episode_time=5.0, init_pos_range=0.5, ref_range=1.5,
                 cost_scale=0.01, smooth_weight=0.05, alive_bonus=0.5,
                 crash_penalty=50.0, tilt_limit_deg=80.0, pos_limit=4.0):
        self.cfg = load_config(config_path)
        self.P_nom = params_from_config(self.cfg)
        self.P = self.P_nom
        tm = self.cfg["timing"]
        self.dt_sim, self.dt_ctrl = tm["dt_sim"], tm["dt_ctrl"]
        self.nsub = n_substeps(self.dt_ctrl, self.dt_sim)
        self.loop_delay = tm["loop_delay"]

        self.task, self.randomize = task, randomize
        self.max_steps = int(round(episode_time / self.dt_ctrl))
        self.init_pos_range, self.ref_range = init_pos_range, ref_range
        self.cost_scale, self.smooth_weight = cost_scale, smooth_weight
        self.alive_bonus, self.crash_penalty = alive_bonus, crash_penalty
        self.tilt_limit, self.pos_limit = np.deg2rad(tilt_limit_deg), pos_limit

        self.Q = np.diag(self.cfg["lqr"]["Q_diag"])
        self.R = np.diag(self.cfg["lqr"]["R_diag"])

        self.action_space = spaces.Box(-1.0, 1.0, shape=(N_U,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(16,), dtype=np.float32)

    # --- mapeo acción <-> empuje -------------------------------------------
    def action_to_thrust(self, a):
        lo, hi = self.P.thrust_min, self.P.thrust_max
        return lo + (np.clip(a, -1, 1) + 1.0) * 0.5 * (hi - lo)

    def thrust_to_action(self, u):
        lo, hi = self.P.thrust_min, self.P.thrust_max
        return 2.0 * (u - lo) / (hi - lo) - 1.0

    # --- Gymnasium API ------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        rc = self.cfg["randomization"]

        if self.randomize:
            self.P = self.P_nom.perturbed(rng, rc)
            delay = rng.uniform(*rc["loop_delay_range"])
            self.noise_std = rc["sensor_noise_std"]
            self.wind = rng.normal(0.0, rc["wind_std"], size=3) * np.array([1, 1, 0])
        else:
            self.P, delay, self.noise_std, self.wind = self.P_nom, self.loop_delay, None, None

        u_hover = np.full(N_U, self.P.hover_thrust)
        self.delay = DelayLine(int(round(delay / self.dt_ctrl)), u_hover)

        if self.task == "track":
            self.ref = np.r_[rng.uniform(-self.ref_range, self.ref_range, 3), 0.0]
        else:
            self.ref = np.zeros(4)

        self.x = reference_state(self.P, np.zeros(4))
        self.x[0:3] += rng.uniform(-self.init_pos_range, self.init_pos_range, 3)
        self.a_prev = self.thrust_to_action(u_hover)
        self.k = 0
        return self._obs(), {}

    def step(self, action):
        a = np.asarray(action, float)
        u = self.delay(self.action_to_thrust(a))
        for _ in range(self.nsub):
            self.x = rk4_step(self.x, u, self.dt_sim, self.P, self.wind)
        self.k += 1

        e = state_error(self.x, reference_state(self.P, self.ref))
        du = u - self.P.hover_thrust
        cost = e @ self.Q @ e + du @ self.R @ du
        smooth = np.sum((a - self.a_prev) ** 2)
        reward = self.alive_bonus - self.cost_scale * cost - self.smooth_weight * smooth
        self.a_prev = a

        crashed = (np.any(np.abs(self.x[6:8]) > self.tilt_limit)
                   or np.linalg.norm(e[0:3]) > self.pos_limit
                   or not np.all(np.isfinite(self.x)))
        if crashed:
            reward -= self.crash_penalty
        truncated = self.k >= self.max_steps
        info = {"cost": cost, "crashed": crashed, "thrust": u}
        return self._obs(), float(reward), bool(crashed), bool(truncated), info

    def _obs(self):
        e = state_error(self.x, reference_state(self.P, self.ref))
        o = np.r_[e[0:3], self.x[3:6], self.x[6:8], e[8], self.x[9:12], self.a_prev]
        if self.noise_std is not None:
            s = self.noise_std
            o[0:12] += self.np_random.normal(0, np.repeat([s["pos"], s["vel"], s["att"], s["rate"]], 3))
        return o.astype(np.float32)
