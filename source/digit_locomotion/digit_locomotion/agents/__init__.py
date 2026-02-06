"""RL agent configurations for Digit locomotion training."""

from .rsl_rl_cfg import (
    DigitFlatPPORunnerCfg,
    DigitRoughPPORunnerCfg,
    DigitMinimalPPORunnerCfg,
    DigitRunningPPORunnerCfg,
)

from .baseline_cfg import (
    DigitBaselineTeacherPPORunnerCfg,
    DigitBaselineStudentPPORunnerCfg,
    DigitBaselineMlpPPORunnerCfg,
    DigitBaselineRecurrentPPORunnerCfg,
)

__all__ = [
    # Standard configs
    "DigitFlatPPORunnerCfg",
    "DigitRoughPPORunnerCfg",
    "DigitMinimalPPORunnerCfg",
    "DigitRunningPPORunnerCfg",
    # Baseline configs (Radosavovic et al. 2024)
    "DigitBaselineTeacherPPORunnerCfg",
    "DigitBaselineStudentPPORunnerCfg",
    "DigitBaselineMlpPPORunnerCfg",
    "DigitBaselineRecurrentPPORunnerCfg",
]
