"""Pruebas de sanidad de la semana 1. Ejecuta: python -m pytest -q"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quadrl.controllers import LQI, LQR, linearize                       # noqa: E402
from quadrl.dynamics import f, hover_state, load_config, params_from_config  # noqa: E402
from quadrl.env import QuadEnv                                           # noqa: E402
from quadrl.sim import simulate                                          # noqa: E402

CFG = load_config()
P = params_from_config(CFG)
TM = CFG["timing"]


def test_hover_is_equilibrium():
    x0 = hover_state(P)
    assert np.allclose(f(x0, np.full(4, P.hover_thrust), P), 0.0, atol=1e-10)


def test_mixer_signs():
    """Más empuje en motores con y>0 -> roll positivo; par de yaw por giro."""
    M = P.mixer
    assert np.isclose(M[0].sum(), 4.0)
    assert np.isclose(M[1].sum(), 0.0, atol=1e-12) and np.isclose(M[2].sum(), 0.0, atol=1e-12)
    assert np.isclose(M[3].sum(), 0.0, atol=1e-12)


def test_linearization_structure():
    A, B = linearize(P)
    g = P.gravity
    # Acoplamientos clásicos de hover: x_dd = g*theta, y_dd = -g*phi
    assert np.isclose(A[3, 7], g, rtol=1e-4) and np.isclose(A[4, 6], -g, rtol=1e-4)
    assert np.allclose(B[12:16], np.eye(4) / P.tau_motor, rtol=1e-4)


def _closed_loop_poles_stable(ctrl, A, B):
    return np.all(np.abs(np.linalg.eigvals(A - B @ ctrl)) < 1.0)


def test_lqr_and_lqi_stable():
    lqr = LQR(P, CFG, TM["dt_ctrl"])
    assert _closed_loop_poles_stable(lqr.K, lqr.Ad, lqr.Bd)
    lqi = LQI(P, CFG, TM["dt_ctrl"])
    assert _closed_loop_poles_stable(np.hstack([lqi.Kx, lqi.Ki]), lqi.Aa, lqi.Ba)


def test_tracking_step_nonlinear():
    ref = np.array(CFG["validation"]["reference"], float)
    for C in (LQR, LQI):
        t, X, _ = simulate(C(P, CFG, TM["dt_ctrl"]), P, hover_state(P), ref, 10.0,
                           TM["dt_sim"], TM["dt_ctrl"])
        assert np.linalg.norm(X[-1, 0:3] - ref[:3]) < 0.05, C.name


def test_env_api_and_lqr_in_env():
    from gymnasium.utils.env_checker import check_env
    env = QuadEnv()
    check_env(env, skip_render_check=True)
    lqr = LQR(P, CFG, TM["dt_ctrl"])
    obs, _ = env.reset(seed=0)
    total, done = 0.0, False
    while not done:
        a = env.thrust_to_action(lqr(env.x, env.ref))
        obs, r, term, trunc, info = env.step(a)
        total += r
        done = term or trunc
    assert not info["crashed"]
    assert np.linalg.norm(env.x[0:3]) < 0.05


def test_randomized_env_runs():
    env = QuadEnv(task="track", randomize=True)
    env.reset(seed=1)
    for _ in range(50):
        env.step(env.action_space.sample())
