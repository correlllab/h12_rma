"""Decoder for extrinsic factors (e_t): reconstructs e_t from latent z."""

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
class EnvFactorDecoderCfg:
    """Configuration for environment factor decoder."""
    latent_dim: int = 8              # input latent dimension
    et_dim: int = 24                 # output e_t dimension (15 joints + 3 torso + 3 left + 3 right)
    dec_hidden_dims: list = None     # decoder hidden layer sizes
    activation: str = 'elu'
    learning_rate: float = 1e-3
    max_grad_norm: float = 10.0
    
    def __post_init__(self):
        if self.dec_hidden_dims is None:
            self.dec_hidden_dims = [256, 256]


class EnvFactorDecoder(nn.Module):
    """Decodes latent representation back to extrinsic factors e_t."""
    
    def __init__(self, cfg: EnvFactorDecoderCfg):
        super().__init__()
        
        self.latent_dim = cfg.latent_dim
        self.et_dim = cfg.et_dim
        self.learning_rate = cfg.learning_rate
        self.max_grad_norm = cfg.max_grad_norm
        
        activation = get_activation(cfg.activation)
        
        # Decoder: z → e_t
        # Input: latent_dim
        # Output: et_dim
        dec_input_dim = self.latent_dim
        dec_layers = []
        for hidden_dim in cfg.dec_hidden_dims:
            dec_layers += [nn.Linear(dec_input_dim, hidden_dim), activation]
            dec_input_dim = hidden_dim
        dec_layers += [nn.Linear(dec_input_dim, self.et_dim)]
        self.decoder = nn.Sequential(*dec_layers)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    
    def forward(self, z):
        """
        Decode latent to e_t.
        
        Args:
            z: (num_envs, latent_dim)
        
        Returns:
            e_t_recon: (num_envs, et_dim)
        """
        return self.decoder(z)
    
    def update(self, z, target_et, lr=None):
        """
        Update decoder to reconstruct e_t from latent.
        
        Args:
            z: (num_envs, latent_dim)
            target_et: (num_envs, et_dim) - true extrinsic factors
            lr: optional new learning rate
        
        Returns:
            loss: float
        """
        if lr is not None:
            self.learning_rate = lr
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.learning_rate
        
        e_t_recon = self.decoder(z)
        
        # MSE loss for e_t reconstruction
        loss = F.mse_loss(e_t_recon, target_et.detach())
        
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        return loss.item()
