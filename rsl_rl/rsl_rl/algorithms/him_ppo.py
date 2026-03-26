"""PPO algorithm with RMA encoder/decoder training."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class HIMPPO:
    """
    PPO with HIM (Hierarchical Imitation Model):
    - Trains policy on actor observations (proprioceptive + optional RMA latent)
    - Trains critic on critic observations (privileged + optional RMA latent)
    - Optionally trains encoder to infer RMA latent
    - Optionally trains decoder to reconstruct extrinsic factors
    """
    
    def __init__(self, actor_critic, device='cpu', encoder=None, decoder=None,
                 learning_rate=1e-3,
                 gamma=0.99,
                 lam=0.95,
                 entropy_coef=0.01,
                 value_coef=1.0,
                 max_grad_norm=1.0,
                 **kwargs):
        
        self.device = device
        self.actor_critic = actor_critic
        self.encoder = encoder
        self.decoder = decoder
        
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.lam = lam
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        
        self.optimizer = torch.optim.Adam(
            self.actor_critic.parameters(),
            lr=self.learning_rate
        )
    
    def init_storage(self, num_envs, num_steps, num_obs, num_critic_obs, num_actions, 
                     env_factors_shape=None):
        """Initialize storage for rollouts."""
        self.num_envs = num_envs
        self.num_steps = num_steps
        
        # Observation storages
        self.obs = []
        self.critic_obs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.log_probs = []
        
        # Environment factors (e_t) if using RMA
        self.env_factors = None
        if env_factors_shape is not None:
            self.env_factors = []
    
    def update(self, obs, critic_obs, actions, rewards, dones, values, 
               next_obs, next_critic_obs, next_value,
               env_factors=None, env_factors_next=None,
               num_learning_epochs=5, num_mini_batches=4):
        """
        Update policy and optional encoder/decoder.
        
        Args:
            obs: (num_steps, num_envs, num_obs)
            critic_obs: (num_steps, num_envs, num_critic_obs)
            actions: (num_steps, num_envs, num_actions)
            rewards: (num_steps, num_envs)
            dones: (num_steps, num_envs)
            values: (num_steps, num_envs)
            next_obs: (num_envs, num_obs)
            next_critic_obs: (num_envs, num_critic_obs)
            next_value: (num_envs,)
            env_factors: (num_steps, num_envs, et_dim) - for encoder/decoder training
            env_factors_next: (num_envs, et_dim)
            num_learning_epochs: PPO epochs
            num_mini_batches: number of mini-batches per epoch
        
        Returns:
            policy_loss, value_loss, entropy_loss, encoder_loss, decoder_loss
        """
        # Compute advantages
        advantages, returns = self._compute_advantages(
            rewards, values, next_value, dones
        )
        
        # Flatten for batch processing
        flat_obs = obs.view(-1, obs.size(-1))
        flat_critic_obs = critic_obs.view(-1, critic_obs.size(-1))
        flat_actions = actions.view(-1, actions.size(-1))
        flat_advantages = advantages.view(-1)
        flat_returns = returns.view(-1)
        
        # Training loop
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy_loss = 0.0
        total_encoder_loss = 0.0
        total_decoder_loss = 0.0
        num_updates = 0
        
        for epoch in range(num_learning_epochs):
            # Shuffle indices
            batch_size = flat_obs.size(0)
            indices = torch.randperm(batch_size, device=self.device)
            
            for i in range(0, batch_size, batch_size // num_mini_batches):
                batch_indices = indices[i:i + batch_size // num_mini_batches]
                
                batch_obs = flat_obs[batch_indices]
                batch_critic_obs = flat_critic_obs[batch_indices]
                batch_actions = flat_actions[batch_indices]
                batch_advantages = flat_advantages[batch_indices]
                batch_returns = flat_returns[batch_indices]
                
                # Policy and value updates
                log_prob, entropy, value = self.actor_critic.evaluate(
                    batch_obs, batch_critic_obs, batch_actions
                )
                
                # Policy loss (PPO)
                ratio = torch.exp(log_prob - log_prob.detach())
                clip_ratio = torch.clamp(ratio, 0.8, 1.2)
                policy_loss = -torch.min(
                    ratio * batch_advantages,
                    clip_ratio * batch_advantages
                ).mean()
                
                # Value loss
                value_loss = F.mse_loss(value, batch_returns)
                
                # Entropy loss
                entropy_loss = -self.entropy_coef * entropy.mean()
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss + entropy_loss
                
                # Optional encoder loss
                encoder_loss = 0.0
                if self.encoder is not None and env_factors is not None:
                    # Extract velocity from next state (assuming it's in privileged obs)
                    # This is a simple version; adapt based on your obs structure
                    encoder_loss = 0.0  # Placeholder
                
                # Optional decoder loss
                decoder_loss = 0.0
                if self.decoder is not None and env_factors is not None:
                    # Decode latent z back to e_t
                    decoder_loss = 0.0  # Placeholder
                
                total_loss = loss + encoder_loss + decoder_loss
                
                # Backward
                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.actor_critic.parameters(),
                    self.max_grad_norm
                )
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_loss.item()
                total_encoder_loss += encoder_loss
                total_decoder_loss += decoder_loss
                num_updates += 1
        
        return {
            'policy_loss': total_policy_loss / num_updates,
            'value_loss': total_value_loss / num_updates,
            'entropy_loss': total_entropy_loss / num_updates,
            'encoder_loss': total_encoder_loss / num_updates,
            'decoder_loss': total_decoder_loss / num_updates,
        }
    
    def _compute_advantages(self, rewards, values, next_value, dones):
        """Compute GAE advantages and returns."""
        num_steps = rewards.size(0)
        num_envs = rewards.size(1)
        
        advantages = torch.zeros(num_steps, num_envs, device=self.device)
        returns = torch.zeros(num_steps, num_envs, device=self.device)
        
        gae = torch.zeros(num_envs, device=self.device)
        next_val = next_value
        
        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                next_non_terminal = 1.0 - dones[t]
                next_val = next_value
            else:
                next_non_terminal = 1.0 - dones[t]
                next_val = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_val * next_non_terminal - values[t]
            gae = delta + self.gamma * self.lam * next_non_terminal * gae
            
            advantages[t] = gae
            returns[t] = gae + values[t]
        
        return advantages, returns
