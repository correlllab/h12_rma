"""Encoder for extrinsic factors (e_t): learns latent representation from observation history."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


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


@dataclass
class EnvFactorEncoderCfg:
    """Configuration for environment factor encoder."""
    temporal_steps: int = 3          # history length for encoder input
    num_one_step_obs: int = 50       # privileged obs dimension
    latent_dim: int = 8              # output latent dimension
    enc_hidden_dims: list = None     # encoder hidden layer sizes
    activation: str = 'elu'
    learning_rate: float = 1e-3
    max_grad_norm: float = 10.0
    
    def __post_init__(self):
        if self.enc_hidden_dims is None:
            self.enc_hidden_dims = [256, 256]


class EnvFactorEncoder(nn.Module):
    """Encodes observation history to latent representation of extrinsic factors."""
    
    def __init__(self, cfg: EnvFactorEncoderCfg = None, **kwargs):
        super().__init__()
        
        # Support both config object and keyword arguments
        if cfg is None:
            cfg = EnvFactorEncoderCfg(**kwargs)
        elif kwargs:
            # If both cfg and kwargs provided, kwargs override cfg values
            for key, value in kwargs.items():
                setattr(cfg, key, value)
        
        self.temporal_steps = cfg.temporal_steps
        self.num_one_step_obs = cfg.num_one_step_obs
        self.latent_dim = cfg.latent_dim
        self.learning_rate = cfg.learning_rate
        self.max_grad_norm = cfg.max_grad_norm
        
        activation = get_activation(cfg.activation)
        
        # Encoder: obs_history → (vel, z)
        # Input: temporal_steps * num_one_step_obs concatenated observations
        enc_input_dim = self.temporal_steps * self.num_one_step_obs
        enc_layers = []
        for hidden_dim in cfg.enc_hidden_dims:
            enc_layers += [nn.Linear(enc_input_dim, hidden_dim), activation]
            enc_input_dim = hidden_dim
        # Output: 3 (velocity) + latent_dim
        enc_layers += [nn.Linear(enc_input_dim, 3 + self.latent_dim)]
        self.encoder = nn.Sequential(*enc_layers)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    
    def forward(self, obs_history):
        """
        Encode observation history to latent.
        
        Args:
            obs_history: (num_envs, temporal_steps * num_one_step_obs) or
                        (num_envs, temporal_steps, num_one_step_obs)
        
        Returns:
            z: (num_envs, latent_dim) - normalized latent representation
        """
        # Flatten if needed
        if obs_history.dim() == 3:
            obs_history = obs_history.view(obs_history.size(0), -1)
        
        parts = self.encoder(obs_history.detach())
        vel, z = parts[..., :3], parts[..., 3:]
        z = F.normalize(z, dim=-1, p=2.0)
        return z.detach()
    
    def encode(self, obs_history):
        """
        Returns both velocity and latent.
        
        Args:
            obs_history: (num_envs, temporal_steps * num_one_step_obs)
        
        Returns:
            vel: (num_envs, 3)
            z: (num_envs, latent_dim)
        """
        parts = self.encoder(obs_history.detach())
        vel, z = parts[..., :3], parts[..., 3:]
        z = F.normalize(z, dim=-1, p=2.0)
        return vel, z
    
    def update(self, obs_history, target_vel, lr=None):
        """
        Update encoder to predict velocity from observation history.
        
        Args:
            obs_history: (num_envs, temporal_steps * num_one_step_obs)
            target_vel: (num_envs, 3) - true velocity to predict
            lr: optional new learning rate
        
        Returns:
            loss: float
        """
        if lr is not None:
            self.learning_rate = lr
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.learning_rate
        
        vel, z = self.encode(obs_history)
        
        # MSE loss for velocity prediction
        loss = F.mse_loss(vel, target_vel.detach())
        
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        return loss.item()
