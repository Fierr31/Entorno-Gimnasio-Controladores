"""Procedencia de resultados: separa lo publicable de lo privado del hardware.

Todo resultado (figura, CSV, política entrenada) se genera con un cfg que
sabe si usó parámetros identificados (cfg["_private"]). Con parámetros
privados, la salida va a rutas con "_private" / runs_private/, que .gitignore
excluye y el hook de pre-commit bloquea, y lleva la marca HW-PRIVATE.
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRIVATE_MARK = "HW-PRIVATE"


def tag(cfg: dict, stem: str) -> str:
    """'validation_lqr' -> 'validation_lqr_private' si se usaron parámetros privados."""
    return f"{stem}_private" if cfg.get("_private") else stem


def run_dir(cfg: dict, name: str) -> Path:
    """Carpeta para una corrida de entrenamiento/evaluación, con su procedencia."""
    from .env import INTERFACE        # import tardío para evitar ciclos

    base = ROOT / ("runs_private" if cfg.get("_private") else "runs")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    d = base / f"{stamp}_{name}"
    d.mkdir(parents=True, exist_ok=True)
    meta = {
        "private": bool(cfg.get("_private")),
        "config_source": cfg.get("_source"),
        "git_commit": _git_commit(),
        "interface": INTERFACE,
        "thrust_limits_N": [cfg["physics"]["thrust_min"], cfg["physics"]["thrust_max"]],
        "dt_ctrl": cfg["timing"]["dt_ctrl"],
        "motors": cfg["motors"],
    }
    if meta["private"]:
        meta["marker"] = PRIVATE_MARK
    (d / "provenance.json").write_text(json.dumps(meta, indent=2))
    return d


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None
