"""Validación del entorno Python contra Simulink (entregable de la semana 1).

Uso:
    python scripts/validate_vs_simulink.py                         # solo Python
    python scripts/validate_vs_simulink.py \
        --simulink data/simulink_lqr.csv --controller LQR \
        --gains data/K_lqr_matlab.csv

CSV de Simulink: columnas  t, x1..x16  (orden de STATE_NAMES), con o sin
encabezado. Usa scripts/export_simulink.m para generarlo.
Criterio de aceptación por defecto: NRMSE < 2 % en posición y actitud.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quadrl.controllers import LQI, LQR                       # noqa: E402
from quadrl.dynamics import STATE_NAMES, load_config, params_from_config  # noqa: E402
from quadrl.provenance import tag                             # noqa: E402
from quadrl.sim import simulate                               # noqa: E402

PANELS = [("Posición [m]", [0, 1, 2]), ("Velocidad [m/s]", [3, 4, 5]),
          ("Actitud [rad]", [6, 7, 8]), ("Vel. angular [rad/s]", [9, 10, 11]),
          ("Empuje motores [N]", [12, 13, 14, 15])]
KEY_STATES = [0, 1, 2, 6, 7, 8]


def load_csv(path):
    data = np.genfromtxt(path, delimiter=",", skip_header=0)
    if np.isnan(data[0]).any():
        data = data[1:]
    if data.shape[1] != 17:
        raise ValueError(f"Se esperaban 17 columnas (t + 16 estados), hay {data.shape[1]}")
    return data[:, 0], data[:, 1:]


def nrmse(a, b):
    rng = np.ptp(b)
    return np.sqrt(np.mean((a - b) ** 2)) / (rng if rng > 1e-9 else 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--public-only", action="store_true",
                    help="ignora hw_identified.yaml (para figuras que se publican)")
    ap.add_argument("--controller", choices=["LQR", "LQI"], default="LQR")
    ap.add_argument("--simulink", help="CSV con t, x1..x16 exportado de Simulink")
    ap.add_argument("--gains", help="CSV con la K de MATLAB (4x16 LQR o 4x20 LQI)")
    ap.add_argument("--tol", type=float, default=0.02)
    args = ap.parse_args()

    cfg = load_config(args.config, private=False if args.public_only else None)
    origin = "PRIVADOS (hardware identificado)" if cfg["_private"] else "públicos (nominales)"
    print(f"[config] parámetros {origin}: {cfg['_source']}")
    P = params_from_config(cfg)
    tm, val = cfg["timing"], cfg["validation"]
    ctrl = (LQR if args.controller == "LQR" else LQI)(P, cfg, tm["dt_ctrl"])

    # 1) Ganancias: si no coinciden, no tiene sentido comparar trayectorias
    if args.gains:
        K_py = ctrl.K if args.controller == "LQR" else np.hstack([ctrl.Kx, ctrl.Ki])
        K_ml = np.loadtxt(args.gains, delimiter=",")
        rel = np.abs(K_py - K_ml).max() / np.abs(K_ml).max()
        print(f"[ganancias] max |K_py - K_matlab| / max|K_matlab| = {rel:.2e}")
        if rel > 1e-3:
            print("  -> Las ganancias difieren: revisa orden de estados/motores, signos del mixer,"
                  " dt_ctrl y Q/R antes de seguir.")

    x0 = np.array([P.hover_thrust if v is None else v for v in val["x0"]], float)
    t, X, U = simulate(ctrl, P, x0, np.array(val["reference"], float), val["t_final"],
                       tm["dt_sim"], tm["dt_ctrl"], tm["loop_delay"])

    out_dir = ROOT / "data"
    np.savetxt(out_dir / f"{tag(cfg, 'python_' + args.controller.lower())}.csv", np.column_stack([t, X]),
               delimiter=",", header="t," + ",".join(STATE_NAMES), comments="")

    sim = None
    if args.simulink:
        ts, Xs = load_csv(args.simulink)
        Xs_i = np.column_stack([np.interp(t, ts, Xs[:, i]) for i in range(16)])
        sim = Xs_i
        print(f"\n[trayectorias] NRMSE Python vs Simulink ({args.controller})")
        worst = 0.0
        for i, n in enumerate(STATE_NAMES):
            e = nrmse(X[:, i], Xs_i[:, i])
            flag = "  <-- revisar" if (i in KEY_STATES and e > args.tol) else ""
            worst = max(worst, e) if i in KEY_STATES else worst
            print(f"  {n:>6s}: {100 * e:6.2f} %{flag}")
        verdict = "PASA" if worst <= args.tol else "NO PASA"
        print(f"\n  Peor NRMSE en posición/actitud: {100 * worst:.2f} %  ->  {verdict} (tol {100 * args.tol:.0f} %)")

    fig, axes = plt.subplots(len(PANELS), 1, figsize=(9, 13), sharex=True)
    for ax, (title, idx) in zip(axes, PANELS):
        for c, i in enumerate(idx):
            col = f"C{c}"
            ax.plot(t, X[:, i], color=col, lw=1.6, label=f"{STATE_NAMES[i]} Python")
            if sim is not None:
                ax.plot(t, sim[:, i], color=col, lw=1.0, ls="--", label=f"{STATE_NAMES[i]} Simulink")
        ax.set_ylabel(title)
        ax.grid(alpha=0.3)
        ax.legend(ncol=2 if sim is None else 4, fontsize=7, loc="best")
    axes[-1].set_xlabel("t [s]")
    src = "Python vs Simulink" if sim is not None else "Python"
    fig.suptitle(f"Validación del entorno — {args.controller}, {src}")
    fig.tight_layout()
    fig_path = ROOT / "figures" / f"{tag(cfg, 'validation_' + args.controller.lower())}.png"
    fig.savefig(fig_path, dpi=150)
    print(f"\nFigura: {fig_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
