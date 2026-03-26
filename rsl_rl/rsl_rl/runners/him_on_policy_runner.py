"""HIM On-Policy Runner: trains policy with optional RMA encoder/decoder."""

import time
import os
from collections import deque
import statistics

import torch
from torch.utils.tensorboard import SummaryWriter

from rsl_rl.algorithms import HIMPPO
from rsl_rl.modules import HIMActorCritic, EnvFactorEncoder, EnvFactorDecoder


class HIMOnPolicyRunner:
    """
    Multi-modal on-policy training with HIM:
    - Actor: proprioceptive observations (+ optional RMA latent history)
    - Critic: privileged observations (+ optional RMA latent)
    - Optional: encoder learns to infer RMA latent z from observation history
    - Optional: decoder reconstructs e_t from z
    """
    
    def __init__(self,
                 env,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):
        
        self.cfg = train_cfg.get("runner", {})
        self.alg_cfg = train_cfg.get("algorithm", {})
        self.policy_cfg = train_cfg.get("policy", {})
        
        self.device = device
        self.env = env
        self.log_dir = log_dir
        
        # Check if RMA is enabled
        self.use_rma = hasattr(env, 'rma_torso') and hasattr(env, 'rma_left') and hasattr(env, 'rma_right')
        
        # Determine observation dimensions
        num_critic_obs = self.env.num_privileged_obs if self.env.num_privileged_obs is not None else self.env.num_obs
        num_one_step_critic_obs = getattr(self.env, 'num_one_step_privileged_obs', self.env.num_one_step_obs)
        self.num_actor_obs = self.env.num_obs
        self.num_critic_obs = num_critic_obs
        
        # RMA dimensions (if enabled)
        self.rma_latent_dim = 0
        self.rma_actor_z_dim = 0
        if self.use_rma:
            self.rma_latent_dim = 8  # encoder output dimension
            self.rma_actor_z_dim = self.rma_latent_dim * 3  # history of 3 steps
            self.num_actor_obs += self.rma_actor_z_dim
            self.num_critic_obs += self.rma_latent_dim
        
        # Create policy
        policy_cfg_dict = dict(self.policy_cfg)
        if self.use_rma:
            policy_cfg_dict['rma_latent_dim'] = self.rma_latent_dim
        
        # Filter policy_cfg_dict to only include valid HIMActorCritic parameters
        valid_keys = {'activation', 'actor_hidden_dims', 'critic_hidden_dims', 'rma_latent_dim'}
        filtered_cfg_dict = {k: v for k, v in policy_cfg_dict.items() if k in valid_keys}
        
        self.policy = HIMActorCritic(
            self.num_actor_obs,
            self.num_critic_obs,
            self.env.num_one_step_obs,
            num_one_step_critic_obs,
            self.env.actor_history_length,
            self.env.critic_history_length,
            self.env.num_lower_dof,
            **filtered_cfg_dict
        ).to(self.device)
        
        # Create encoder and decoder (optional)
        self.encoder = None
        self.decoder = None
        self._build_et_from_gym = None
        if self.use_rma:
            from rma_modules import build_et_from_gym
            self._build_et_from_gym = build_et_from_gym
            
            # Encoder: obs_history → latent z
            self.encoder = EnvFactorEncoder(
                temporal_steps=3,
                num_one_step_obs=num_one_step_critic_obs,
                latent_dim=self.rma_latent_dim,
            ).to(self.device)
            
            # Decoder: latent z → e_t
            self.decoder = EnvFactorDecoder(
                latent_dim=self.rma_latent_dim,
                et_dim=24,  # 15 joints + 3 torso + 3 left + 3 right
            ).to(self.device)
            
            # Latent history for actor
            self._z_history = torch.zeros(
                self.env.num_envs,
                3,  # history length
                self.rma_latent_dim,
                device=self.device
            )
        
        # Create algorithm
        alg_cfg_dict = dict(self.alg_cfg)
        self.alg = HIMPPO(
            self.policy,
            device=self.device,
            encoder=self.encoder,
            decoder=self.decoder,
            **alg_cfg_dict
        )
        
        # Initialize storage
        env_factors_shape = (24,) if self.use_rma else None
        self.alg.init_storage(
            self.env.num_envs,
            self.cfg.get("num_steps_per_env", 10),
            [self.num_actor_obs],
            [self.num_critic_obs],
            [self.env.num_lower_dof],
            env_factors_shape=env_factors_shape
        )
        
        # Logging
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        
        # Reset environment
        _, _ = self.env.reset()
    
    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        """Main training loop."""
        
        # Initialize logger
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf,
                high=int(self.env.max_episode_length)
            )
        
        # Get initial observations
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        
        # Initialize RMA (if enabled)
        if self.use_rma and self._build_et_from_gym is not None and self.encoder is not None:
            e_t0 = self._build_et_from_gym(
                self.env.dof_pos,
                self.env.rma_torso,
                self.env.rma_left,
                self.env.rma_right,
                self.env.dof_names,
            ).to(self.device)
            z_t0 = self.encoder(e_t0)
            critic_obs = torch.cat([critic_obs, z_t0], dim=-1)
            
            # Initialize z_history: newest at index 0
            self._z_history.zero_()
            self._z_history[:, 0, :] = z_t0
        
        self.policy.train()
        
        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        
        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        
        for it in range(start_iter, tot_iter):
            start = time.time()
            
            # Collect rollouts
            with torch.inference_mode():
                for step in range(self.cfg.get("num_steps_per_env", 10)):
                    
                    # Update encoder/decoder (if RMA)
                    if self.use_rma and self._build_et_from_gym is not None and self.encoder is not None:
                        e_t = self._build_et_from_gym(
                            self.env.dof_pos,
                            self.env.rma_torso,
                            self.env.rma_left,
                            self.env.rma_right,
                            self.env.dof_names,
                        ).to(self.device)
                        z_t = self.encoder(e_t)
                        critic_obs = torch.cat([privileged_obs.to(self.device), z_t], dim=-1)
                        
                        # Update z_history: newest at index 0 (chronological for actor)
                        self._z_history[:, 1:, :] = self._z_history[:, :-1, :].clone()
                        self._z_history[:, 0, :] = z_t
                    
                    # Get actions from policy
                    if self.use_rma and self._z_history is not None:
                        # Concatenate z_history with obs for actor
                        z_history_flat = self._z_history.view(self.env.num_envs, -1)
                        actor_obs_input = torch.cat([obs, z_history_flat], dim=-1)
                    else:
                        actor_obs_input = obs
                    
                    actions = self.policy.act(actor_obs_input, critic_obs)
                    
                    # Step environment
                    obs, _, rews, dones, infos = self.env.step(actions)
                    obs = obs.to(self.device)
                    
                    # Update buffers
                    cur_reward_sum += rews
                    cur_episode_length += 1
                    
                    # Log episode info on reset
                    done_ids = dones.nonzero(as_tuple=False)
                    if len(done_ids) > 0:
                        for i in done_ids:
                            ep_infos.append({
                                'reward': cur_reward_sum[i].item(),
                                'length': cur_episode_length[i].item(),
                            })
                            rewbuffer.append(cur_reward_sum[i].item())
                            lenbuffer.append(cur_episode_length[i].item())
                        
                        cur_reward_sum[done_ids] = 0
                        cur_episode_length[done_ids] = 0
            
            # Get privileged obs for critic
            if self.env.num_privileged_obs is not None:
                privileged_obs = self.env.get_privileged_observations()
                critic_obs = privileged_obs.to(self.device)
            else:
                critic_obs = obs.to(self.device)
            
            # Log statistics
            if self.log_dir is not None and self.writer is not None:
                if len(rewbuffer) > 0:
                    self.writer.add_scalar('Episode/Reward_Mean', statistics.mean(rewbuffer), it)
                    self.writer.add_scalar('Episode/Length_Mean', statistics.mean(lenbuffer), it)
                
                # Log RMA forces (if enabled)
                if self.use_rma:
                    self.writer.add_scalar(
                        'RMA/torso_force_l2_mean',
                        torch.norm(self.env.rma_torso, dim=1).mean().item(),
                        it
                    )
                    self.writer.add_scalar(
                        'RMA/left_hand_force_l2_mean',
                        torch.norm(self.env.rma_left, dim=1).mean().item(),
                        it
                    )
                    self.writer.add_scalar(
                        'RMA/right_hand_force_l2_mean',
                        torch.norm(self.env.rma_right, dim=1).mean().item(),
                        it
                    )
            
            # Time
            iteration_time = time.time() - start
            self.tot_time += iteration_time
            self.tot_timesteps += self.env.num_envs * self.cfg.get("num_steps_per_env", 10)
            self.current_learning_iteration += 1
    
    def save(self, path):
        """Save policy checkpoint."""
        checkpoint = {
            'policy': self.policy.state_dict(),
            'encoder': self.encoder.state_dict() if self.encoder is not None else None,
            'decoder': self.decoder.state_dict() if self.decoder is not None else None,
        }
        torch.save(checkpoint, path)
    
    def load(self, path):
        """Load policy checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy'])
        if self.encoder is not None and checkpoint['encoder'] is not None:
            self.encoder.load_state_dict(checkpoint['encoder'])
        if self.decoder is not None and checkpoint['decoder'] is not None:
            self.decoder.load_state_dict(checkpoint['decoder'])
