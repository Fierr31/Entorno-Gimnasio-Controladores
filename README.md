# Learned vs. optimal control for a quadrotor: PPO vs. LQR/LQI

Work in progress. A PPO policy trained on the same 16-state quadrotor model
(rigid body + first-order motor dynamics) used in my COMRob 2026 paper,
compared against discrete LQR and LQI under identical references,
disturbances and loop delay. The RL reward is the LQR quadratic cost with
the same Q and R, so both controllers optimize the same objective: one
with an explicit model, one learned.

## Structure

```
config/quad_params.yaml      physical params, timing, Q/R, validation scenario
config/hw_identified.example.yaml   template for local overrides (git-ignored copy)
quadrl/dynamics.py           nonlinear 16-state model, RK4
quadrl/controllers.py        linearization, ZOH discretization, LQR, LQI (anti-windup)
quadrl/sim.py                closed-loop sim with ZOH and loop-delay buffer
quadrl/env.py                Gymnasium env (per-motor thrust commands) + INTERFACE
quadrl/provenance.py         tags outputs and runs with the parameter source
scripts/validate_vs_simulink.py   Python vs Simulink overlay + NRMSE
scripts/export_simulink.m    exports Simulink logs and gains to CSV
tests/                       sanity tests
```

## Quick start

```bash
pip install -r requirements.txt
sh scripts/setup_hooks.sh        # once per clone
python -m pytest -q
python scripts/validate_vs_simulink.py --controller LQR \
    --simulink data/simulink_lqr.csv --gains data/K_lqr_matlab.csv
```

## Parameters

`config/quad_params.yaml` holds nominal parameters for a generic F450-class
quadrotor; every published figure and policy is produced with them
(`--public-only`). Hardware-identified parameters are kept out of this
repository.

## Policy interface

`quadrl.env.INTERFACE` defines the observation layout, units, frame
conventions and action-to-thrust mapping. Each training run stores it in
`provenance.json` so any runtime can check compatibility. On-board
deployment is out of scope for this repository.

## Status

- [x] Python model, LQR/LQI, Gymnasium env
- [ ] Environment validated against Simulink
- [ ] PPO hover
- [ ] Reference tracking with domain randomization
- [ ] RL vs LQR vs LQI benchmark
