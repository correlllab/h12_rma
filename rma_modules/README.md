# RMA (Rapid Motor Adaptation) for H12 Locomotion

This directory contains the RMA (Rapid Motor Adaptation) implementation for H1-2 robot control with **torso and hand force sampling**.

## Overview

**RMA** trains a policy that adapts to arbitrary extrinsic factors (forces) applied to the robot during training. The key idea:
- During training, sample various forces (torso and wrists) and apply them to the simulation
- The policy learns to adapt to these forces in-context, without retraining
- The "extrinsic vector" **e_t** (what the robot doesn't directly observe) captures these forces

## What Gets Sampled?

Our **e_t** is 24-dimensional:
- **15 upper-body joint positions** (read from `dof_pos` each step)
- **3 torso forces** (Fx, Fy, Fz): uniformly sampled from ±30 N per axis
- **3 left hand forces** (Fx, Fy, Fz): spherically sampled, magnitude 0–30 N
- **3 right hand forces** (Fx, Fy, Fz): spherically sampled, magnitude 0–30 N

This design allows the policy to learn robust locomotion under arbitrary external pushes to the torso and wrist impacts.

## File Structure

### Core Modules

**`env_factor_spec.py`** — Specification of the extrinsic vector layout and force ranges
- `RmaEtSpec`: Dataclass defining e_t structure (24-dim)
- Force ranges: `TORSO_FORCE_RANGE` (±30), `HAND_FORCE_MAGNITUDE_RANGE` (0–30)
- Resample probability: `RMA_RESAMPLE_PROB = 0.004` per step (RMA paper)
- `UPPER_BODY_JOINT_NAMES`: 15 joint names for filling e_t from `dof_pos`
- `RMA_FORCE_BODY_NAMES`: 3 body names for applying forces to simulation

**`gym_et_builder.py`** — Force sampling and e_t construction
- `sample_rma_forces(num_envs, device)`: Sample torso (uniform) and hand (spherical) forces
  - Returns: `(torso, left_hand, right_hand)` each shape `(num_envs, 3)`
- `resample_rma_forces_for_envs(...)`: In-place resample for specified env indices
- `build_et_from_gym(dof_pos, torso, left, right, dof_names)`: Build 24-dim e_t from joint positions and forces
- `make_rma_force_tensor(...)`: Create (num_envs, num_bodies, 3) force tensor for Isaac Gym

**`__init__.py`** — Package exports and documentation

### Integration with Legged Gym

**`legged_gym/envs/base/legged_robot_rma.py`** — RMA-enabled environment
- `LeggedRobotRMA`: Subclass of `LeggedRobot` that:
  - Samples initial torso/hand forces at reset
  - Resamples forces each step with probability `RMA_RESAMPLE_PROB`
  - Applies forces to simulation via `gym.apply_rigid_body_force_tensors()`
  - Exposes `self.rma_torso`, `self.rma_left`, `self.rma_right` for logging/debugging

**`legged_gym/envs/h1_2/h1_2_rma_env.py`** — H1-2 specific RMA environment
- `H1_2RMARobot`: Subclass of `LeggedRobotRMA` with H1-2 observation/reward structure
- Inherits all RMA force sampling from `LeggedRobotRMA`

**`legged_gym/envs/h1_2/h1_2_rma_config.py`** — Configuration for RMA training
- `H1_2RMACfg`: Environment config (from `H1_2RoughCfg` base)
- `H1_2RMACfgPPO`: PPO hyperparameters for H1-2 RMA training

## How It Works

### 1. Initialization (at episode reset)

```python
env = LeggedRobotRMA(cfg, ...)  # in __init__
# Samples initial forces
torso, left, right = sample_rma_forces(num_envs, device)
env.rma_torso[:] = torso
env.rma_left[:] = left
env.rma_right[:] = right
```

### 2. Per-Step Resampling (in step())

```python
# With probability 0.004 per env, resample
resample_mask = torch.rand(num_envs) < RMA_RESAMPLE_PROB
resample_env_ids = resample_mask.nonzero(as_tuple=False).flatten()
resample_rma_forces_for_envs(env.rma_torso, env.rma_left, env.rma_right, resample_env_ids)

# Build force tensor and apply to simulation
force_tensor = make_rma_force_tensor(..., env.rma_torso, env.rma_left, env.rma_right, ...)
gym.apply_rigid_body_force_tensors(sim, force_tensor, None, ENV_SPACE)
```

### 3. Building Extrinsic Vector e_t

Each training step, the runner can build e_t:

```python
e_t = build_et_from_gym(dof_pos, env.rma_torso, env.rma_left, env.rma_right, dof_names)
# e_t shape: (num_envs, 24)
```

This e_t can be logged, stored for offline training, or passed to an encoder.

## Training Usage

### Option 1: Policy Only (Base Adaptation)

Train a policy that takes standard observations plus the extrinsic vector e_t:

```python
# In runner or policy:
obs = env.obs_buf  # shape (num_envs, 47)
e_t = build_et_from_gym(dof_pos, env.rma_torso, env.rma_left, env.rma_right, dof_names)  # (num_envs, 24)
if use_rma:
    policy_input = torch.cat([obs, e_t], dim=-1)  # (num_envs, 47 + 24 = 71)
    actions = policy(policy_input)
```

### Option 2: With Encoder (Full RMA)

Train an encoder network to infer e_t from observations and use it:

```python
# Encoder infers extrinsics from observation history
obs_history = storage.observations[t-H:t]  # last H steps
z = encoder(obs_history)  # learned latent representation
...
# Decoder reconstructs e_t for loss
e_t_recon = decoder(z)
loss = MSE(e_t_recon, e_t_true)
```

See `homie_h12/HomieRL` for a full encoder/decoder implementation (HIMEstimator).

## Configuration

To train H1-2 with RMA:

```bash
cd legged_gym
python scripts/train.py --task h1_2_rma
```

This uses:
- Config: `h1_2_rma_config.H1_2RMACfgPPO`
- Environment: `h1_2_rma_env.H1_2RMARobot`
- Force ranges: 
  - Torso: ±30 N per axis
  - Hands: 0–30 N magnitude, uniform direction

## Logging

The runner logs RMA forces:

```python
# In runner step:
writer.add_scalar('RMA/torso_force_l2_mean', torch.norm(env.rma_torso, dim=1).mean().item(), it)
writer.add_scalar('RMA/left_hand_force_l2_mean', torch.norm(env.rma_left, dim=1).mean().item(), it)
writer.add_scalar('RMA/right_hand_force_l2_mean', torch.norm(env.rma_right, dim=1).mean().item(), it)
```

## Deployment

After training a policy on RMA forces, deploy with **zero external forces** by setting all force components to zero:

```python
# At deployment:
e_t_deployment = torch.cat([upper_body_dofs, torch.zeros(3, device=device), torch.zeros(3, device=device), torch.zeros(3, device=device)], dim=-1)
actions = policy(torch.cat([obs, e_t_deployment], dim=-1))
```

The policy generalizes to the nominal (zero-force) case despite training on arbitrary forces.

## Key Parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| Torso force range | ±30 N | `TORSO_FORCE_RANGE` |
| Hand force magnitude | 0–30 N | `HAND_FORCE_MAGNITUDE_RANGE` |
| Resample probability | 0.004 per step | `RMA_RESAMPLE_PROB` |
| e_t dimension | 24 | `RmaEtSpec.dim` |
| Upper-body joints | 15 | `UPPER_BODY_JOINT_NAMES` |

Adjust these in `rma_modules/env_factor_spec.py` and redeploy if needed.

## References

- RMA paper: "Rapid Motor Adaptation for Legged Robots" (Lewis et al., 2023)
- HomieRL implementation: See `homie_h12/HomieRL/` and `homie_h12/RMA/`

## Troubleshooting

**Forces not being applied?**
- Check that body names match: `torso_link`, `left_wrist_roll_link`, `right_wrist_roll_link`
- Verify `LeggedRobotRMA._before_simulate()` is called (should be automatic in base class)
- Check Isaac Gym API (e.g., `gymapi.ENV_SPACE` for world frame)

**Wrong force dimensions in e_t?**
- Ensure `UPPER_BODY_JOINT_NAMES` matches 15 joints in your URDF
- Verify `build_et_from_gym()` concatenation order: upper (15) + torso (3) + left (3) + right (3)

**Encoder training diverging?**
- Start with `learning_rate = 1e-3` and `num_learning_epochs = 5` (see config)
- Use gradient clipping and layer normalization in encoder
- See `homie_h12/HomieRL` for a tested encoder architecture
