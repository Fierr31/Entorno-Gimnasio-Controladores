"""Pruebas de la frontera público/privado del hardware. Ejecuta: python -m pytest -q"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quadrl.dynamics import load_config, params_from_config   # noqa: E402
from quadrl.provenance import PRIVATE_MARK, run_dir, tag      # noqa: E402


@pytest.fixture
def private_file(tmp_path):
    p = tmp_path / "hw_identified.yaml"
    p.write_text(yaml.safe_dump({"physics": {"mass": 1.75, "inertia": None},
                                 "lqr": {"R_diag": [0.5, 0.5, 0.5, 0.5]}}))
    return p


def test_override_merges_only_given_keys(private_file):
    pub = load_config(private=False)
    cfg = load_config(private=private_file)
    assert cfg["_private"] and not pub["_private"]
    assert cfg["physics"]["mass"] == 1.75
    assert cfg["physics"]["inertia"] == pub["physics"]["inertia"]     # null conserva el público
    assert cfg["physics"]["arm_length"] == pub["physics"]["arm_length"]
    assert cfg["lqr"]["R_diag"] == [0.5] * 4
    assert params_from_config(cfg).mass == 1.75


def test_env_var_override(private_file, monkeypatch):
    monkeypatch.setenv("QUADRL_PRIVATE_CONFIG", str(private_file))
    assert load_config()["_private"]
    assert not load_config(private=False)["_private"]


def test_private_outputs_are_tagged(private_file, tmp_path, monkeypatch):
    import quadrl.provenance as prov
    monkeypatch.setattr(prov, "ROOT", tmp_path)
    cfg = load_config(private=private_file)
    assert tag(cfg, "validation_lqr") == "validation_lqr_private"
    d = run_dir(cfg, "ppo_hover")
    assert d.parent.name == "runs_private"
    meta = json.loads((d / "provenance.json").read_text())
    assert meta["private"] and meta["marker"] == PRIVATE_MARK and meta["interface"]["version"] == 1

    pub = load_config(private=False)
    assert tag(pub, "validation_lqr") == "validation_lqr"
    assert run_dir(pub, "ppo_hover").parent.name == "runs"


@pytest.mark.skipif(shutil.which("git") is None, reason="git no disponible")
def test_pre_commit_hook_blocks_private(tmp_path):
    repo = tmp_path / "repo"
    shutil.copytree(ROOT / ".githooks", repo / ".githooks")
    (repo / "config").mkdir()
    (repo / "data").mkdir()
    run = lambda *a: subprocess.run(a, cwd=repo, capture_output=True, text=True)
    run("git", "init", "-q")
    run("git", "config", "core.hooksPath", ".githooks")
    run("git", "config", "user.email", "t@t")
    run("git", "config", "user.name", "t")

    def try_commit(rel, content="x: 1\n"):
        (repo / rel).write_text(content)
        run("git", "add", "-f", rel)
        r = run("git", "commit", "-qm", "t")
        run("git", "reset", "-q", "--hard") if r.returncode else None
        return r.returncode == 0

    assert try_commit("config/hw_identified.example.yaml")
    assert not try_commit("config/hw_identified.yaml")
    assert not try_commit("data/K_lqr_matlab.csv")
    assert not try_commit("figures_validation_lqr_private.png")
    assert not try_commit("notes.json", f'{{"marker": "{PRIVATE_MARK}"}}\n')
    assert try_commit("README.md", "# público\n")
