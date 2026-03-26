"""Build RMA e_t from Isaac Gym and apply sampled forces to simulation.

Torso + Hand RMA: 3D torso forces (uniform per axis) + 3D hand forces (spherical sampling).
- e_t: 15 upper-body from dof_pos + torso_xyz(3) + left_hand_xyz(3) + right_hand_xyz(3) = 24.
- Torso: Fx, Fy, Fz each U(±30).
- Hands: Magnitude U(0, 30), direction = normalize(randn(3)).
- Resample with probability 0.004 per step.
- Requires: torso_link, left_wrist_roll_link, right_wrist_roll_link.
"""

from __future__ import annotations

import torch

from .env_factor_spec import (
    DEFAULT_ET_SPEC,
    HAND_FORCE_MAGNITUDE_RANGE,
    TORSO_FORCE_RANGE,
    RmaEtSpec,
    UPPER_BODY_JOINT_NAMES,
)


def _sample_direction_spherical(n: int, device: torch.device) -> torch.Tensor:
    """Unit-norm direction: Gaussian then normalize (uniform on sphere). Shape (n, 3)."""
    d = torch.randn(n, 3, device=device, dtype=torch.float32)
    norm = torch.norm(d, dim=1, keepdim=True).clamp(min=1e-6)
    return d / norm


def sample_rma_forces(
    num_envs: int,
    device: torch.device,
    et_spec: RmaEtSpec | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample torso and hand forces.

    Torso: 3D uniform, each component U(TORSO_FORCE_RANGE).
    Hands: 3D spherical sampling (magnitude then direction).

    Returns:
        torso_force: (num_envs, 3) in world frame (N)
        left_hand_force: (num_envs, 3) in world frame (N)
        right_hand_force: (num_envs, 3) in world frame (N)
    """
    if et_spec is None:
        et_spec = DEFAULT_ET_SPEC
    
    # Torso: uniform per axis
    torso_lo, torso_hi = TORSO_FORCE_RANGE
    torso = torch.rand(num_envs, 3, device=device, dtype=torch.float32) * (torso_hi - torso_lo) + torso_lo
    
    # Hands: spherical sampling
    lo, hi = HAND_FORCE_MAGNITUDE_RANGE
    mag_left = torch.rand(num_envs, 1, device=device, dtype=torch.float32) * (hi - lo) + lo
    mag_right = torch.rand(num_envs, 1, device=device, dtype=torch.float32) * (hi - lo) + lo
    dir_left = _sample_direction_spherical(num_envs, device)
    dir_right = _sample_direction_spherical(num_envs, device)
    left = mag_left * dir_left
    right = mag_right * dir_right
    
    return torso, left, right


def resample_rma_forces_for_envs(
    torso_force: torch.Tensor,
    left_hand_force: torch.Tensor,
    right_hand_force: torch.Tensor,
    env_ids: torch.Tensor,
    et_spec: RmaEtSpec | None = None,
) -> None:
    """In-place: resample torso and hand forces for the given env indices."""
    if env_ids.numel() == 0:
        return
    device = torso_force.device
    n = env_ids.shape[0]
    torso, left, right = sample_rma_forces(n, device, et_spec)
    torso_force[env_ids] = torso
    left_hand_force[env_ids] = left
    right_hand_force[env_ids] = right


def build_et_from_gym(
    dof_pos: torch.Tensor,
    torso_force: torch.Tensor,
    left_hand_force: torch.Tensor,
    right_hand_force: torch.Tensor,
    dof_names: list[str],
    et_spec: RmaEtSpec | None = None,
) -> torch.Tensor:
    """Build e_t: 15 upper-body from dof_pos + torso_xyz(3) + left_xyz(3) + right_xyz(3) = 24.

    Args:
        dof_pos: (num_envs, num_dof)
        torso_force: (num_envs, 3)
        left_hand_force: (num_envs, 3)
        right_hand_force: (num_envs, 3)
        dof_names: from gym.get_asset_dof_names
        et_spec: optional spec

    Returns:
        e_t: (num_envs, 24)
    """
    if et_spec is None:
        et_spec = DEFAULT_ET_SPEC
    upper_indices = [dof_names.index(name) for name in UPPER_BODY_JOINT_NAMES]
    upper = dof_pos[:, upper_indices]
    e_t = torch.cat([upper, torso_force, left_hand_force, right_hand_force], dim=-1)
    return e_t


def make_rma_force_tensor(
    num_envs: int,
    num_bodies: int,
    torso_body_index: int,
    left_wrist_body_index: int,
    right_wrist_body_index: int,
    torso_force: torch.Tensor,
    left_hand_force: torch.Tensor,
    right_hand_force: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Fill (num_envs, num_bodies, 3) force tensor. Torso and hands each get 3D force.

    Tensor is in ENV_SPACE (world frame). Call:
        gym.apply_rigid_body_force_tensors(sim, gymtorch.unwrap_tensor(forces), None, gymapi.ENV_SPACE)
    """
    forces = torch.zeros((num_envs, num_bodies, 3), device=device, dtype=torch.float32)
    forces[:, torso_body_index, :] = torso_force
    forces[:, left_wrist_body_index, :] = left_hand_force
    forces[:, right_wrist_body_index, :] = right_hand_force
    return forces
