"""RSL-RL: Fast Reinforcement Learning Library for H12 RMA."""

__version__ = "1.0.0"

from rsl_rl.modules import (
    HIMActorCritic,
    EnvFactorEncoder,
    EnvFactorEncoderCfg,
    EnvFactorDecoder,
    EnvFactorDecoderCfg,
)

from rsl_rl.algorithms import HIMPPO

from rsl_rl.runners import HIMOnPolicyRunner

__all__ = [
    'HIMActorCritic',
    'EnvFactorEncoder',
    'EnvFactorEncoderCfg',
    'EnvFactorDecoder',
    'EnvFactorDecoderCfg',
    'HIMPPO',
    'HIMOnPolicyRunner',
]
