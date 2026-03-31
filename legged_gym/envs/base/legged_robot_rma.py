# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from __future__ import annotations

import os
import sys

import torch

from isaacgym import gymapi, gymtorch

from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg


def _import_rma():
    try:
        from rma_modules import (
            RMA_RESAMPLE_PROB,
            make_rma_force_tensor,
            resample_rma_forces_for_envs,
            sample_rma_forces,
        )
        return sample_rma_forces, resample_rma_forces_for_envs, make_rma_force_tensor, RMA_RESAMPLE_PROB
    except ImportError:
        from legged_gym import LEGGED_GYM_ROOT_DIR
        repo_root = os.path.abspath(os.path.join(LEGGED_GYM_ROOT_DIR, ".."))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from rma_modules import (
            RMA_RESAMPLE_PROB,
            make_rma_force_tensor,
            resample_rma_forces_for_envs,
            sample_rma_forces,
        )
        return sample_rma_forces, resample_rma_forces_for_envs, make_rma_force_tensor, RMA_RESAMPLE_PROB


class LeggedRobotRMA(LeggedRobot):
    """RMA with torso + hand 3D forces: torso uniform per axis, hands spherical sampling, applied to torso and wrists.

    - At reset: sample torso and hand forces for reset envs.
    - Each step: with probability RMA_RESAMPLE_PROB per env, resample; then apply
      rma_torso (num_envs, 3), rma_left (num_envs, 3), and rma_right (num_envs, 3) 
      to torso_link, left_wrist_yaw_link, and right_wrist_yaw_link.
    - Requires: torso_link, left_wrist_yaw_link, right_wrist_yaw_link.
    """

    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

        (
            self._rma_sample_forces,
            self._rma_resample_for_envs,
            self._rma_make_force_tensor,
            self._rma_resample_prob,
        ) = _import_rma()

        # Find body indices for RMA force application
        self._rma_torso_body_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], "torso_link")
        self._rma_left_wrist_body_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], "left_wrist_yaw_link")
        self._rma_right_wrist_body_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], "right_wrist_yaw_link")

        # RMA-required observation attributes for HIMOnPolicyRunner
        self.num_one_step_obs = self.cfg.env.num_observations
        self.num_one_step_privileged_obs = self.cfg.env.num_privileged_obs
        self.actor_history_length = 3
        self.critic_history_length = 3
        self.num_lower_dof = 6  # number of leg DOFs (3 per leg)

        # (num_envs, 3) each — 3D force in world frame
        self.rma_torso = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float32)
        self.rma_left = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float32)
        self.rma_right = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float32)

        torso, left, right = self._rma_sample_forces(self.num_envs, self.device)
        self.rma_torso[:] = torso
        self.rma_left[:] = left
        self.rma_right[:] = right

        self._rma_force_tensor = self._rma_make_force_tensor(
            self.num_envs,
            self.num_bodies,
            self._rma_torso_body_index,
            self._rma_left_wrist_body_index,
            self._rma_right_wrist_body_index,
            self.rma_torso,
            self.rma_left,
            self.rma_right,
            self.device,
        )

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if len(env_ids) == 0:
            return
        torso, left, right = self._rma_sample_forces(len(env_ids), self.device)
        self.rma_torso[env_ids] = torso
        self.rma_left[env_ids] = left
        self.rma_right[env_ids] = right

    def step(self, actions):
        resample_mask = torch.rand(
            self.num_envs, device=self.device, dtype=torch.float32
        ) < self._rma_resample_prob
        resample_env_ids = resample_mask.nonzero(as_tuple=False).flatten()
        self._rma_resample_for_envs(
            self.rma_torso, self.rma_left, self.rma_right, resample_env_ids
        )

        self._rma_force_tensor = self._rma_make_force_tensor(
            self.num_envs,
            self.num_bodies,
            self._rma_torso_body_index,
            self._rma_left_wrist_body_index,
            self._rma_right_wrist_body_index,
            self.rma_torso,
            self.rma_left,
            self.rma_right,
            self.device,
        )

        return super().step(actions)

    def _before_simulate(self):
        # RMA: apply 3D torso and hand forces. Logged in runner as RMA/torso_*, RMA/left_hand_*, RMA/right_hand_*.
        self.gym.apply_rigid_body_force_tensors(
            self.sim,
            gymtorch.unwrap_tensor(self._rma_force_tensor),
            None,
            gymapi.ENV_SPACE,
        )
