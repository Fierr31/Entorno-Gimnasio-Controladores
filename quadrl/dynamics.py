"""Modelo no lineal de 16 estados del cuadricóptero con dinámica de motores.

Estado x (16):
    0:3   p      posición inercial [m] (z hacia arriba)
    3:6   v      velocidad inercial [m/s]
    6:9   eta    ángulos de Euler ZYX (phi, theta, psi) [rad]
    9:12  omega  velocidades angulares en cuerpo (p, q, r) [rad/s]
    12:16 T      empuje real de cada motor [N]
Entrada u (4): empuje comandado por motor [N]; el motor es un primer orden
    dT/dt = (u_sat - T) / tau_motor.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import yaml

N_X, N_U = 16, 4
STATE_NAMES = ["x", "y", "z", "vx", "vy", "vz", "phi", "theta", "psi",
               "p", "q", "r", "T1", "T2", "T3", "T4"]


@dataclass
class QuadParams:
    mass: float
    gravity: float
    inertia: np.ndarray
    arm_length: float
    c_tau: float
    tau_motor: float
    thrust_max: float
    thrust_min: float
    linear_drag: np.ndarray
    angles_deg: np.ndarray
    spin: np.ndarray
    mixer: np.ndarray = field(init=False)   # [F; Mx; My; Mz] = mixer @ T

    def __post_init__(self):
        self.inertia = np.asarray(self.inertia, float)
        self.linear_drag = np.asarray(self.linear_drag, float)
        self.angles_deg = np.asarray(self.angles_deg, float)
        self.spin = np.asarray(self.spin, float)
        a = np.deg2rad(self.angles_deg)
        px, py = self.arm_length * np.cos(a), self.arm_length * np.sin(a)
        # r x F con F = T e3:  Mx = y*T,  My = -x*T;  Mz por par de reacción
        self.mixer = np.vstack([np.ones(4), py, -px, self.spin * self.c_tau])

    @property
    def J(self) -> np.ndarray:
        return np.diag(self.inertia)

    @property
    def hover_thrust(self) -> float:
        return self.mass * self.gravity / 4.0

    def perturbed(self, rng: np.random.Generator, cfg: dict) -> "QuadParams":
        """Copia con parámetros aleatorizados (factores uniformes ±)."""
        f = lambda k: 1.0 + rng.uniform(-cfg[k], cfg[k])
        return replace(self, mass=self.mass * f("mass"),
                       inertia=self.inertia * np.array([f("inertia") for _ in range(3)]),
                       tau_motor=self.tau_motor * f("tau_motor"))


CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"
PUBLIC_CONFIG = CONFIG_DIR / "quad_params.yaml"
PRIVATE_CONFIG = CONFIG_DIR / "hw_identified.yaml"   # nunca se versiona (.gitignore + hook)


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        elif v is not None:                 # null en el override = conservar el valor público
            out[k] = v
    return out


def load_config(path: str | Path = None, private: str | Path | bool | None = None) -> dict:
    """Carga la configuración pública y, si existe, la sobreescribe con la privada.

    private:
        None  -> usa $QUADRL_PRIVATE_CONFIG o config/hw_identified.yaml si existen
        False -> solo parámetros públicos (usar para todo lo que se publique)
        ruta  -> usa ese archivo privado
    cfg["_private"] indica si se usaron parámetros identificados; los scripts de
    entrenamiento lo usan para decidir si un resultado puede publicarse.
    """
    with open(Path(path or PUBLIC_CONFIG)) as fh:
        cfg = yaml.safe_load(fh)

    if private is False:
        priv_path = None
    elif private is None:
        env = os.environ.get("QUADRL_PRIVATE_CONFIG")
        priv_path = Path(env) if env else PRIVATE_CONFIG
    else:
        priv_path = Path(private)

    cfg["_private"] = False
    if priv_path is not None and priv_path.exists():
        with open(priv_path) as fh:
            override = yaml.safe_load(fh) or {}
        cfg = _deep_merge(cfg, override)
        cfg["_private"] = True
    cfg["_source"] = str(priv_path) if cfg["_private"] else str(Path(path or PUBLIC_CONFIG))
    return cfg


def params_from_config(cfg: dict) -> QuadParams:
    return QuadParams(**cfg["physics"], **cfg["motors"])


def rotation(eta: np.ndarray) -> np.ndarray:
    """R_body→inercial, convención ZYX."""
    phi, th, psi = eta
    cf, sf, ct, st, cp, sp = np.cos(phi), np.sin(phi), np.cos(th), np.sin(th), np.cos(psi), np.sin(psi)
    return np.array([
        [ct * cp, sf * st * cp - cf * sp, cf * st * cp + sf * sp],
        [ct * sp, sf * st * sp + cf * cp, cf * st * sp - sf * cp],
        [-st,     sf * ct,                cf * ct],
    ])


def euler_rates_matrix(eta: np.ndarray) -> np.ndarray:
    """W(eta) tal que eta_dot = W @ omega."""
    phi, th, _ = eta
    cf, sf, ct, tt = np.cos(phi), np.sin(phi), np.cos(th), np.tan(th)
    return np.array([[1, sf * tt, cf * tt], [0, cf, -sf], [0, sf / ct, cf / ct]])


def f(x: np.ndarray, u: np.ndarray, P: QuadParams, f_ext: np.ndarray | None = None) -> np.ndarray:
    """dx/dt del modelo no lineal. f_ext: fuerza externa inercial (viento) [N]."""
    v, eta, w, T = x[3:6], x[6:9], x[9:12], x[12:16]
    u = np.clip(u, P.thrust_min, P.thrust_max)

    wrench = P.mixer @ T
    F, M = wrench[0], wrench[1:]
    R = rotation(eta)

    acc = R @ np.array([0.0, 0.0, F]) / P.mass - np.array([0.0, 0.0, P.gravity]) - P.linear_drag * v / P.mass
    if f_ext is not None:
        acc = acc + f_ext / P.mass

    J = P.J
    w_dot = np.linalg.solve(J, M - np.cross(w, J @ w))

    dx = np.empty(N_X)
    dx[0:3] = v
    dx[3:6] = acc
    dx[6:9] = euler_rates_matrix(eta) @ w
    dx[9:12] = w_dot
    dx[12:16] = (u - T) / P.tau_motor
    return dx


def rk4_step(x, u, dt, P, f_ext=None):
    k1 = f(x, u, P, f_ext)
    k2 = f(x + 0.5 * dt * k1, u, P, f_ext)
    k3 = f(x + 0.5 * dt * k2, u, P, f_ext)
    k4 = f(x + dt * k3, u, P, f_ext)
    return x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


def hover_state(P: QuadParams, pos=(0.0, 0.0, 0.0)) -> np.ndarray:
    x = np.zeros(N_X)
    x[0:3] = pos
    x[12:16] = P.hover_thrust
    return x
