"""Locomotion tasks for Digit robot."""

import gymnasium as gym

from . import mdp
from .velocity_env_cfg import (
    DigitFlatEnvCfg,
    DigitRoughEnvCfg,
    DigitMinimalEnvCfg,
    # Curriculum phase configs
    DigitWalkingEnvCfg,
    DigitJoggingWarmupEnvCfg,
    DigitJoggingIntroEnvCfg,
    DigitJoggingIntroScheduledEnvCfg,  # With reward scheduling
    DigitJoggingEnvCfg,
    DigitRunningEnvCfg,
    DigitFastRunningEnvCfg,
    DigitSprintEnvCfg,
)

# Baseline environments (Radosavovic et al. 2024 approach)
from .baseline_env_cfg import (
    DigitBaselineEnvCfg,
    DigitBaselineTeacherEnvCfg,
    DigitBaselineFastWalkingEnvCfg,
    DigitBaselineJoggingEnvCfg,
    DigitBaselineRunningEnvCfg,
    DigitBaselineFastRunningEnvCfg,
)

# =============================================================================
# BASE ENVIRONMENTS
# =============================================================================

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

gym.register(
    id="Digit-Velocity-Minimal-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:DigitMinimalEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__.rsplit('.', 2)[0]}.agents.rsl_rl_cfg:DigitMinimalPPORunnerCfg",
    },
)

# =============================================================================
# CURRICULUM PHASE ENVIRONMENTS (30 mph target)
# =============================================================================
# Use these in sequence, resuming from checkpoint each time:
#   Phase 1:    Digit-Walking-v0       (0-2 m/s)   - Learn balance (upright)
#   Phase 1.5:  Digit-JoggingWarmup-v0 (0-2 m/s)   - Learn jogging posture at walking speed
#   Phase 1.75: Digit-JoggingIntro-v0  (0-3 m/s)   - Bridge to faster jogging
#   Phase 2:    Digit-Jogging-v0       (0-5 m/s)   - Transition to running
#   Phase 3:    Digit-Running-v0       (0-8 m/s)   - Fast running
#   Phase 4:    Digit-FastRunning-v0   (0-10 m/s)  - Sprint warmup
#   Phase 5:    Digit-Sprint-v0        (0-13.5 m/s) - 30 mph sprint

gym.register(
    id="Digit-Walking-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:DigitWalkingEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__.rsplit('.', 2)[0]}.agents.rsl_rl_cfg:DigitRunningPPORunnerCfg",
    },
)

gym.register(
    id="Digit-JoggingWarmup-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:DigitJoggingWarmupEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__.rsplit('.', 2)[0]}.agents.rsl_rl_cfg:DigitRunningPPORunnerCfg",
    },
)

gym.register(
    id="Digit-JoggingIntro-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:DigitJoggingIntroEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__.rsplit('.', 2)[0]}.agents.rsl_rl_cfg:DigitRunningPPORunnerCfg",
    },
)

# Scheduled version - use with --reward_schedule jogging
gym.register(
    id="Digit-JoggingIntroScheduled-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:DigitJoggingIntroScheduledEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__.rsplit('.', 2)[0]}.agents.rsl_rl_cfg:DigitRunningPPORunnerCfg",
    },
)

gym.register(
    id="Digit-Jogging-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:DigitJoggingEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__.rsplit('.', 2)[0]}.agents.rsl_rl_cfg:DigitRunningPPORunnerCfg",
    },
)

gym.register(
    id="Digit-Running-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:DigitRunningEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__.rsplit('.', 2)[0]}.agents.rsl_rl_cfg:DigitRunningPPORunnerCfg",
    },
)

gym.register(
    id="Digit-FastRunning-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:DigitFastRunningEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__.rsplit('.', 2)[0]}.agents.rsl_rl_cfg:DigitRunningPPORunnerCfg",
    },
)

gym.register(
    id="Digit-Sprint-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:DigitSprintEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__.rsplit('.', 2)[0]}.agents.rsl_rl_cfg:DigitRunningPPORunnerCfg",
    },
)

# Keep old registration for backwards compatibility
gym.register(
    id="Digit-Velocity-Running-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:DigitRunningEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__.rsplit('.', 2)[0]}.agents.rsl_rl_cfg:DigitRunningPPORunnerCfg",
    },
)

# =============================================================================
# BASELINE ENVIRONMENTS (Radosavovic et al. 2024)
# =============================================================================
# These follow the Science Robotics paper approach:
#   - NO explicit arm swing rewards (emergent behavior)
#   - Energy minimization for natural motion
#   - Transformer policy with observation history (student)
#   - Teacher-student training pipeline
#
# Training pipeline:
#   1. Train teacher: Digit-BaselineTeacher-v0 (MLP, privileged state)
#   2. Train student: Digit-Baseline-v0 (Transformer, noisy obs)

gym.register(
    id="Digit-Baseline-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.baseline_env_cfg:DigitBaselineEnvCfg",
        # Use same architecture as teacher [512,512,256,128] so weights can be loaded for distillation
        "rsl_rl_cfg_entry_point": f"{__name__.rsplit('.', 2)[0]}.agents.baseline_cfg:DigitBaselineTeacherPPORunnerCfg",
    },
)

gym.register(
    id="Digit-BaselineTeacher-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.baseline_env_cfg:DigitBaselineTeacherEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__.rsplit('.', 2)[0]}.agents.baseline_cfg:DigitBaselineTeacherPPORunnerCfg",
    },
)

# LSTM baseline for comparison (paper shows transformer >> LSTM)
gym.register(
    id="Digit-BaselineLSTM-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.baseline_env_cfg:DigitBaselineEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__.rsplit('.', 2)[0]}.agents.baseline_cfg:DigitBaselineRecurrentPPORunnerCfg",
    },
)

# Baseline curriculum: Fast Walking (0-2 m/s) - Intermediate step
gym.register(
    id="Digit-BaselineFastWalking-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.baseline_env_cfg:DigitBaselineFastWalkingEnvCfg",
        # Same architecture as teacher/student for checkpoint compatibility
        "rsl_rl_cfg_entry_point": f"{__name__.rsplit('.', 2)[0]}.agents.baseline_cfg:DigitBaselineTeacherPPORunnerCfg",
    },
)

# Baseline curriculum: Jogging (0-3 m/s)
gym.register(
    id="Digit-BaselineJogging-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.baseline_env_cfg:DigitBaselineJoggingEnvCfg",
        # Same architecture as teacher/student for checkpoint compatibility
        "rsl_rl_cfg_entry_point": f"{__name__.rsplit('.', 2)[0]}.agents.baseline_cfg:DigitBaselineTeacherPPORunnerCfg",
    },
)

# Baseline curriculum: Running (0-5 m/s)
gym.register(
    id="Digit-BaselineRunning-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.baseline_env_cfg:DigitBaselineRunningEnvCfg",
        # Same architecture as teacher/student for checkpoint compatibility
        "rsl_rl_cfg_entry_point": f"{__name__.rsplit('.', 2)[0]}.agents.baseline_cfg:DigitBaselineTeacherPPORunnerCfg",
    },
)

# Baseline curriculum: Fast Running (0-8 m/s, ~18 mph)
gym.register(
    id="Digit-BaselineFastRunning-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.baseline_env_cfg:DigitBaselineFastRunningEnvCfg",
        # Same architecture as teacher/student for checkpoint compatibility
        "rsl_rl_cfg_entry_point": f"{__name__.rsplit('.', 2)[0]}.agents.baseline_cfg:DigitBaselineTeacherPPORunnerCfg",
    },
)

__all__ = [
    "mdp",
    "DigitFlatEnvCfg",
    "DigitRoughEnvCfg",
    "DigitMinimalEnvCfg",
    "DigitWalkingEnvCfg",
    "DigitJoggingWarmupEnvCfg",
    "DigitJoggingIntroEnvCfg",
    "DigitJoggingIntroScheduledEnvCfg",
    "DigitJoggingEnvCfg",
    "DigitRunningEnvCfg",
    "DigitFastRunningEnvCfg",
    "DigitSprintEnvCfg",
    # Baseline (Radosavovic et al. 2024)
    "DigitBaselineEnvCfg",
    "DigitBaselineTeacherEnvCfg",
    "DigitBaselineFastWalkingEnvCfg",
    "DigitBaselineJoggingEnvCfg",
    "DigitBaselineRunningEnvCfg",
    "DigitBaselineFastRunningEnvCfg",
]
