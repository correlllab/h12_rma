<div align="center">
  <h1 align="center">H1-2 RMA Locomotion</h1>
  <p align="center">
    Rapid Motor Adaptation for Unitree H1-2 with torso and hand force perturbations
  </p>
</div>

<p align="center">
  <strong>Reinforcement learning with RMA (Rapid Motor Adaptation) for robust H1-2 locomotion under arbitrary external forces applied to the torso and wrists.</strong>
</p>

---

## Overview

This repository extends [unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym) with an **RMA training pipeline** for the Unitree H1-2 robot. The key addition is an extrinsic vector **e_t** that captures external forces and upper-body joint states, allowing the policy to adapt in-context without retraining.

### What is RMA?

During training, random forces are sampled and applied to the robot's torso and wrists each step. The policy receives the standard observations plus a 24-dimensional **extrinsic vector e_t** containing:

| Component | Dims | Description |
|-----------|------|-------------|
| Upper-body joint positions | 15 | Read from `dof_pos` each step |
| Torso force (Fx, Fy, Fz) | 3 | Uniformly sampled ±30 N per axis |
| Left hand force (Fx, Fy, Fz) | 3 | Spherically sampled, 0–30 N |
| Right hand force (Fx, Fy, Fz) | 3 | Spherically sampled, 0–30 N |

At deployment, forces are set to zero — the policy generalizes to the nominal case.

---

## Project Structure

```
h12_rma/
├── rma_modules/                   # RMA core logic
│   ├── env_factor_spec.py         # e_t layout, force ranges, resample prob
│   ├── gym_et_builder.py          # Force sampling and e_t construction
│   └── __init__.py
├── legged_gym/
│   ├── envs/
│   │   ├── base/
│   │   │   └── legged_robot_rma.py      # RMA base env (force apply/resample)
│   │   └── h1_2/
│   │       ├── h1_2_rma_env.py          # H1-2 RMA environment
│   │       └── h1_2_rma_config.py       # Env + PPO config for RMA training
│   └── scripts/
│       ├── train.py
│       └── play.py
├── deploy/
│   ├── deploy_mujoco/             # Sim2Sim with Mujoco
│   └── deploy_real/               # Sim2Real deployment
└── rsl_rl/                        # PPO implementation
```

---

## Installation

Please refer to [setup.md](/doc/setup_en.md) for installation steps.

---

## Workflow

`Train` → `Play` → `Sim2Sim` → `Sim2Real`

- **Train**: Run PPO with RMA force perturbations applied each step.
- **Play**: Visualize and verify the trained policy; exports the actor network.
- **Sim2Sim**: Transfer to Mujoco to validate cross-simulator robustness.
- **Sim2Real**: Deploy to the physical H1-2 robot.

---

## User Guide

### 1. Training

#### Standard H1-2 training (no RMA)

```bash
python legged_gym/scripts/train.py --task=h1_2
```

#### H1-2 with RMA

```bash
python legged_gym/scripts/train.py --task=h1_2_rma
```

This uses:
- Environment: `H1_2RMARobot` (`legged_gym/envs/h1_2/h1_2_rma_env.py`)
- Config: `H1_2RMACfgPPO` (`legged_gym/envs/h1_2/h1_2_rma_config.py`)
- Force ranges: torso ±30 N, hands 0–30 N, resample prob 0.004/step

#### Parameters

- `--headless`: Headless mode (higher efficiency).
- `--resume`: Resume from a checkpoint in `logs/`.
- `--experiment_name`: Name of the experiment.
- `--run_name`: Name of the run.
- `--load_run`: Run to load (defaults to latest).
- `--checkpoint`: Checkpoint number (defaults to latest).
- `--num_envs`: Number of parallel environments.
- `--seed`: Random seed.
- `--max_iterations`: Maximum training iterations.
- `--sim_device`: Simulation device (e.g. `--sim_device=cpu`).
- `--rl_device`: RL computation device.

**Output**: `logs/<experiment_name>/<date_time>_<run_name>/model_<iteration>.pt`

---

### 2. Play

```bash
python legged_gym/scripts/play.py --task=h1_2_rma
```

By default loads the latest model from the last run. Specify with `--load_run` and `--checkpoint`.

#### Export Network

Play exports the actor to `logs/{experiment_name}/exported/policies/`:
- MLP policy → `policy_1.pt`
- RNN policy → `policy_lstm_1.pt`

---

### 3. Sim2Sim (Mujoco)

```bash
python deploy/deploy_mujoco/deploy_mujoco.py h1_2.yaml
```

Config files are in `deploy/deploy_mujoco/configs/`. Update `policy_path` in the YAML to point to your exported model; default is `deploy/pre_train/h1_2/motion.pt`.

---

### 4. Sim2Real (Physical Deployment)

Ensure the robot is in debug mode first. See [Physical Deployment Guide](deploy/deploy_real/README.md).

```bash
python deploy/deploy_real/deploy_real.py {net_interface} h1_2.yaml
```

- `net_interface`: Network interface connected to the robot (e.g. `enp3s0`).
- Config files are in `deploy/deploy_real/configs/`.

#### Zero-force deployment

At inference, set all force components of e_t to zero:

```python
e_t = torch.cat([
    upper_body_dofs,          # (num_envs, 15)
    torch.zeros(num_envs, 3), # torso
    torch.zeros(num_envs, 3), # left hand
    torch.zeros(num_envs, 3), # right hand
], dim=-1)  # (num_envs, 24)
actions = policy(torch.cat([obs, e_t], dim=-1))
```

---

## RMA Modules

### `rma_modules/env_factor_spec.py`

Defines the e_t layout and force ranges:

| Symbol | Value |
|--------|-------|
| `TORSO_FORCE_RANGE` | ±30 N |
| `HAND_FORCE_MAGNITUDE_RANGE` | 0–30 N |
| `RMA_RESAMPLE_PROB` | 0.004 per step |
| `RmaEtSpec.dim` | 24 |
| `UPPER_BODY_JOINT_NAMES` | 15 joints |
| `RMA_FORCE_BODY_NAMES` | `torso_link`, `left_wrist_roll_link`, `right_wrist_roll_link` |

### `rma_modules/gym_et_builder.py`

Key functions:

```python
# Sample new forces for all envs
torso, left, right = sample_rma_forces(num_envs, device)

# Resample forces for specific envs (used at reset and per-step)
resample_rma_forces_for_envs(rma_torso, rma_left, rma_right, env_ids)

# Build the 24-dim e_t vector
e_t = build_et_from_gym(dof_pos, rma_torso, rma_left, rma_right, dof_names)

# Build Isaac Gym force tensor for apply_rigid_body_force_tensors
force_tensor = make_rma_force_tensor(num_envs, num_bodies, body_ids, rma_torso, rma_left, rma_right, device)
```

### `legged_gym/envs/base/legged_robot_rma.py`

`LeggedRobotRMA` extends `LeggedRobot`:
- Samples initial forces at reset
- Resamples per-step with probability `RMA_RESAMPLE_PROB`
- Applies forces to simulation via `gym.apply_rigid_body_force_tensors()`
- Exposes `self.rma_torso`, `self.rma_left`, `self.rma_right` for logging

---

## Training with an Encoder (Full RMA)

For full RMA, train an encoder to infer e_t from observation history:

```python
obs_history = storage.observations[t-H:t]  # last H steps
z = encoder(obs_history)                    # latent representation
e_t_recon = decoder(z)
loss = F.mse_loss(e_t_recon, e_t_true)
```

See `homie_h12/HomieRL` for a tested HIMEstimator encoder/decoder architecture.

---

## TensorBoard Logging

The runner logs RMA force magnitudes during training:

```
RMA/torso_force_l2_mean
RMA/left_hand_force_l2_mean
RMA/right_hand_force_l2_mean
```

---

## Troubleshooting

**Forces not being applied?**
- Confirm body names match: `torso_link`, `left_wrist_roll_link`, `right_wrist_roll_link`
- Verify `LeggedRobotRMA._before_simulate()` is called each step
- Check that `gymapi.ENV_SPACE` is used (world frame)

**Wrong e_t dimensions?**
- Ensure `UPPER_BODY_JOINT_NAMES` lists exactly 15 joints matching your URDF
- Verify concatenation order in `build_et_from_gym`: upper (15) + torso (3) + left (3) + right (3)

**Encoder training diverging?**
- Start with `learning_rate = 1e-3` and `num_learning_epochs = 5`
- Use gradient clipping and layer normalization in the encoder

---

## Acknowledgments

- [legged\_gym](https://github.com/leggedrobotics/legged_gym): Training framework foundation
- [rsl\_rl](https://github.com/leggedrobotics/rsl_rl.git): PPO implementation
- [mujoco](https://github.com/google-deepmind/mujoco.git): Sim2Sim simulator
- [unitree\_sdk2\_python](https://github.com/unitreerobotics/unitree_sdk2_python.git): Hardware interface
- RMA paper: "Rapid Motor Adaptation for Legged Robots" (Kumar et al., 2021)

---

## License

This project is licensed under the [BSD 3-Clause License](./LICENSE).
