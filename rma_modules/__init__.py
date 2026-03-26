"""RMA (Rapid Motor Adaptation) modules for H12 with torso + hand forces.

This package provides force sampling, environment factor building, and force application for RMA training.

**Core Workflow:**
1. **Sampling** (`gym_et_builder.sample_rma_forces`): Sample torso (uniform ±30) and hand (spherical) forces.
2. **Building e_t** (`gym_et_builder.build_et_from_gym`): Combine upper-body joint positions with sampled forces.
3. **Applying Forces** (`gym_et_builder.make_rma_force_tensor`): Create force tensor for Isaac Gym.
4. **Environment** (`legged_gym.envs.base.legged_robot_rma.LeggedRobotRMA`): Resample and apply forces each step.

**Specification** (`env_factor_spec`):
- `RmaEtSpec`: Dataclass specifying e_t layout (24-dim: 15 upper-body + 3 torso + 3 left hand + 3 right hand).
- `TORSO_FORCE_RANGE`: ±30 N per axis (uniform).
- `HAND_FORCE_MAGNITUDE_RANGE`: 0–30 N (magnitude, then spherical direction).
- `RMA_RESAMPLE_PROB`: 0.004 per step (RMA paper).

**Example Usage:**
```python
from rma_modules import sample_rma_forces, build_et_from_gym, make_rma_force_tensor, RMA_RESAMPLE_PROB
from rma_modules.env_factor_spec import DEFAULT_ET_SPEC

# Sample forces for all envs
torso, left, right = sample_rma_forces(num_envs, device)

# Build e_t (24-dim extrinsic vector)
e_t = build_et_from_gym(dof_pos, torso, left, right, dof_names)

# Create force tensor and apply to simulation
force_tensor = make_rma_force_tensor(num_envs, num_bodies, torso_idx, left_idx, right_idx, torso, left, right, device)
gym.apply_rigid_body_force_tensors(sim, gymtorch.unwrap_tensor(force_tensor), None, gymapi.ENV_SPACE)
```

**Training Integration:**
- Use `LeggedRobotRMA` (subclass of `LeggedRobot`) to handle force sampling and application.
- Access sampled forces via `env.rma_torso`, `env.rma_left`, `env.rma_right`.
- Resample probability: `RMA_RESAMPLE_PROB` (resampled with this prob each env each step).
"""

from __future__ import annotations

from .env_factor_spec import (
    DEFAULT_ET_SPEC,
    HAND_FORCE_COMPONENT_RANGE,
    HAND_FORCE_MAGNITUDE_RANGE,
    RMA_FORCE_BODY_NAMES,
    RMA_RESAMPLE_PROB,
    TORSO_FORCE_RANGE,
    UPPER_BODY_JOINT_NAMES,
    RmaEtSpec,
)
from .gym_et_builder import (
    build_et_from_gym,
    make_rma_force_tensor,
    resample_rma_forces_for_envs,
    sample_rma_forces,
)

__all__ = [
    "DEFAULT_ET_SPEC",
    "HAND_FORCE_COMPONENT_RANGE",
    "HAND_FORCE_MAGNITUDE_RANGE",
    "RMA_FORCE_BODY_NAMES",
    "RMA_RESAMPLE_PROB",
    "TORSO_FORCE_RANGE",
    "UPPER_BODY_JOINT_NAMES",
    "RmaEtSpec",
    "build_et_from_gym",
    "make_rma_force_tensor",
    "resample_rma_forces_for_envs",
    "sample_rma_forces",
]
