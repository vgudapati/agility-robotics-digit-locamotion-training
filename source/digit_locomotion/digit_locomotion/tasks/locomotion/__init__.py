"""Locomotion tasks for Digit robot."""

import gymnasium as gym

from . import mdp
from .velocity_env_cfg import (
    DigitFlatEnvCfg,
    DigitRoughEnvCfg,
)

# Register Gymnasium environments
gym.register(
    id="Digit-Velocity-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:DigitFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__.rsplit('.', 2)[0]}.agents.rsl_rl_cfg:DigitFlatPPORunnerCfg",
    },
)

gym.register(
    id="Digit-Velocity-Rough-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:DigitRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__.rsplit('.', 2)[0]}.agents.rsl_rl_cfg:DigitRoughPPORunnerCfg",
    },
)

__all__ = [
    "mdp",
    "DigitFlatEnvCfg",
    "DigitRoughEnvCfg",
]
