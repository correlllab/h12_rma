from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

from legged_gym.envs.h1_2.h1_2_config import H1_2RoughCfg, H1_2RoughCfgPPO
from legged_gym.envs.h1_2.h1_2_env import H1_2Robot
from legged_gym.envs.h1_2.h1_2_rma_config import H1_2RMACfg, H1_2RMACfgPPO
from legged_gym.envs.h1_2.h1_2_rma_env import H1_2RMARobot

from .base.legged_robot import LeggedRobot
from .base.legged_robot_rma import LeggedRobotRMA

from legged_gym.utils.task_registry import task_registry

task_registry.register( "h1_2", H1_2Robot, H1_2RoughCfg(), H1_2RoughCfgPPO())
task_registry.register( "h1_2_rma", H1_2RMARobot, H1_2RMACfg(), H1_2RMACfgPPO())
