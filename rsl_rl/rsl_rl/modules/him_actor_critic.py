"""Actor-Critic network for H12 RMA training."""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        raise ValueError(f"activation {act_name} not supported")


class HIMActorCritic(nn.Module):
    """
    Actor-Critic network for hierarchical imitation with multiple modalities.
    
    - Actor: proprioceptive observations (+ optional RMA latent history)
    - Critic: privileged observations (+ optional RMA latent)
    """
    
    def __init__(self,
                 num_actor_obs,
                 num_critic_obs,
                 num_one_step_obs,
                 num_one_step_critic_obs,
                 actor_history_length,
                 critic_history_length,
                 num_actions,
                 activation='elu',
                 actor_hidden_dims=[256, 256, 128],
                 critic_hidden_dims=[256, 256, 128],
                 rma_latent_dim=0,
                 **kwargs):
        
        super().__init__()
        
        if kwargs:
            print(f"HIMActorCritic got unexpected arguments: {list(kwargs.keys())}")
        
        self.num_actor_obs = num_actor_obs
        self.num_critic_obs = num_critic_obs
        self.num_one_step_obs = num_one_step_obs
        self.num_one_step_critic_obs = num_one_step_critic_obs
        self.actor_history_length = actor_history_length
        self.critic_history_length = critic_history_length
        self.num_actions = num_actions
        self.rma_latent_dim = rma_latent_dim
        
        # Actor network
        actor_layers = []
        actor_input_dim = num_actor_obs
        for hidden_dim in actor_hidden_dims:
            actor_layers += [nn.Linear(actor_input_dim, hidden_dim), get_activation(activation)]
            actor_input_dim = hidden_dim
        actor_layers += [nn.Linear(actor_input_dim, num_actions)]
        self.actor_trunk = nn.Sequential(*actor_layers)
        self.actor_mean = nn.Identity()  # mean is output of trunk
        self.actor_std = nn.Parameter(torch.zeros(num_actions))

        # Critic network
        critic_layers = []
        critic_input_dim = num_critic_obs
        for hidden_dim in critic_hidden_dims:
            critic_layers += [nn.Linear(critic_input_dim, hidden_dim), get_activation(activation)]
            critic_input_dim = hidden_dim
        critic_layers += [nn.Linear(critic_input_dim, 1)]
        self.critic_trunk = nn.Sequential(*critic_layers)
        self.critic_value = nn.Identity()
        
        # Initialize std to small value
        nn.init.constant_(self.actor_std, -1.0)
    
    def forward(self, actor_obs, critic_obs):
        """
        Forward pass for both actor and critic.
        
        Args:
            actor_obs: (num_envs, num_actor_obs) - proprioceptive + optional RMA latent
            critic_obs: (num_envs, num_critic_obs) - privileged + optional RMA latent
        
        Returns:
            mean: (num_envs, num_actions)
            std: (num_envs, num_actions)
            value: (num_envs, 1)
        """
        mean = self.actor_trunk(actor_obs)
        value = self.critic_trunk(critic_obs)
        std = torch.exp(self.actor_std)
        return mean, std, value
    
    def act(self, actor_obs, critic_obs):
        """Sample action from policy."""
        mean, std, value = self.forward(actor_obs, critic_obs)
        dist = Normal(mean, std)
        action = dist.sample()
        return action.detach()

    def act_and_log_prob(self, actor_obs, critic_obs):
        """Sample action and return (action, log_prob, value) — all detached."""
        mean, std, value = self.forward(actor_obs, critic_obs)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action.detach(), log_prob.detach(), value.squeeze(-1).detach()
    
    def get_actions_log_prob(self, actor_obs, critic_obs, actions):
        """Get log probability of actions."""
        mean, std, _ = self.forward(actor_obs, critic_obs)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy
    
    def get_value(self, critic_obs):
        """Get value estimate."""
        return self.critic_trunk(critic_obs)
    
    def evaluate(self, actor_obs, critic_obs, actions):
        """
        Evaluate actions for PPO update.
        
        Returns:
            log_prob: (num_envs,)
            entropy: (num_envs,)
            value: (num_envs,)
        """
        mean, std, value = self.forward(actor_obs, critic_obs)
        value = value.squeeze(-1)
        
        dist = Normal(mean, std)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return log_prob, entropy, value
