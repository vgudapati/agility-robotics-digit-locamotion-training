"""Reward scheduler for gradual reward introduction during training.

This module provides a scheduler that allows reward weights to be gradually
introduced during training, enabling locomotion to stabilize before arm
posture constraints are applied.

Example usage:
    scheduler = RewardScheduler(
        schedules={
            "arm_close_to_body": RewardSchedule(
                start_weight=0.0,
                end_weight=0.25,
                start_iteration=200,
                end_iteration=500,
            ),
            "arm_lateral_penalty": RewardSchedule(
                start_weight=0.0,
                end_weight=-0.5,
                start_iteration=200,
                end_iteration=500,
            ),
        }
    )

    # In training loop callback:
    scheduler.update(env, current_iteration)
"""

from dataclasses import dataclass, field
from typing import Dict, Optional
import math


@dataclass
class RewardSchedule:
    """Schedule for a single reward weight.

    The weight transitions from start_weight to end_weight between
    start_iteration and end_iteration using the specified interpolation.

    Attributes:
        start_weight: Initial weight value (before start_iteration)
        end_weight: Final weight value (after end_iteration)
        start_iteration: Iteration at which to begin transitioning
        end_iteration: Iteration at which transition completes
        interpolation: How to interpolate ("linear", "cosine", "step")
    """
    start_weight: float = 0.0
    end_weight: float = 1.0
    start_iteration: int = 0
    end_iteration: int = 500
    interpolation: str = "linear"  # "linear", "cosine", "step"

    def get_weight(self, iteration: int) -> float:
        """Get the weight value for the current iteration."""
        if iteration < self.start_iteration:
            return self.start_weight
        elif iteration >= self.end_iteration:
            return self.end_weight

        # Calculate progress (0 to 1)
        progress = (iteration - self.start_iteration) / (self.end_iteration - self.start_iteration)

        if self.interpolation == "step":
            return self.end_weight if progress >= 0.5 else self.start_weight
        elif self.interpolation == "cosine":
            # Smooth cosine interpolation
            progress = 0.5 * (1 - math.cos(math.pi * progress))
        # else linear - progress is already linear

        return self.start_weight + progress * (self.end_weight - self.start_weight)


@dataclass
class RewardScheduler:
    """Manages scheduled reward weight transitions during training.

    This scheduler allows gradual introduction of reward terms, which is
    useful when training locomotion policies that need to first learn
    balance before being constrained by arm posture requirements.

    Attributes:
        schedules: Dict mapping reward term names to their schedules
        verbose: Whether to print when weights are updated
    """
    schedules: Dict[str, RewardSchedule] = field(default_factory=dict)
    verbose: bool = True
    _last_printed_iteration: int = -100  # For rate-limiting prints

    def get_current_weights(self, iteration: int) -> Dict[str, float]:
        """Get all scheduled weights for the current iteration."""
        return {
            name: schedule.get_weight(iteration)
            for name, schedule in self.schedules.items()
        }

    def update(self, env, iteration: int) -> bool:
        """Update reward weights in the environment.

        Args:
            env: The Isaac Lab environment (unwrapped)
            iteration: Current training iteration

        Returns:
            True if any weights were updated, False otherwise
        """
        updated = False
        current_weights = self.get_current_weights(iteration)

        # Access the reward manager
        reward_manager = env.reward_manager

        for term_name, target_weight in current_weights.items():
            # Check if this reward term exists
            if term_name in reward_manager._term_names:
                # Get the term index
                term_idx = reward_manager._term_names.index(term_name)
                current_weight = reward_manager._term_cfgs[term_idx].weight

                # Only update if weight has changed significantly
                if abs(current_weight - target_weight) > 1e-6:
                    reward_manager._term_cfgs[term_idx].weight = target_weight
                    updated = True

        # Print status periodically (every 50 iterations)
        if self.verbose and updated and iteration - self._last_printed_iteration >= 50:
            self._print_status(iteration, current_weights)
            self._last_printed_iteration = iteration

        return updated

    def _print_status(self, iteration: int, weights: Dict[str, float]):
        """Print current scheduled weights."""
        print(f"\n[RewardScheduler] Iteration {iteration}:")
        for name, weight in sorted(weights.items()):
            schedule = self.schedules[name]
            progress = 0.0
            if iteration >= schedule.end_iteration:
                progress = 100.0
            elif iteration > schedule.start_iteration:
                progress = 100.0 * (iteration - schedule.start_iteration) / (schedule.end_iteration - schedule.start_iteration)
            print(f"  {name}: {weight:.4f} ({progress:.0f}% of transition)")
        print()

    def is_complete(self, iteration: int) -> bool:
        """Check if all scheduled transitions are complete."""
        return all(
            iteration >= schedule.end_iteration
            for schedule in self.schedules.values()
        )


# === Pre-configured schedulers ===

def create_jogging_scheduler(
    locomotion_warmup: int = 200,
    arm_transition_duration: int = 300,
) -> RewardScheduler:
    """Create a scheduler for jogging training.

    This scheduler:
    1. Starts with zero arm posture rewards to let locomotion stabilize
    2. Gradually introduces arm posture constraints after warmup period

    Args:
        locomotion_warmup: Iterations before introducing arm rewards
        arm_transition_duration: Iterations over which to ramp up arm rewards

    Returns:
        Configured RewardScheduler
    """
    arm_start = locomotion_warmup
    arm_end = locomotion_warmup + arm_transition_duration

    return RewardScheduler(
        schedules={
            # Arm posture rewards - start at 0, ramp up gradually
            "arm_close_to_body": RewardSchedule(
                start_weight=0.0,
                end_weight=0.25,
                start_iteration=arm_start,
                end_iteration=arm_end,
                interpolation="cosine",
            ),
            "arm_lateral_penalty": RewardSchedule(
                start_weight=0.0,
                end_weight=-0.5,
                start_iteration=arm_start,
                end_iteration=arm_end,
                interpolation="cosine",
            ),
            # Elbow bend - introduce slightly later
            "elbow_bend": RewardSchedule(
                start_weight=0.0,
                end_weight=0.15,
                start_iteration=arm_start + 100,
                end_iteration=arm_end + 100,
                interpolation="cosine",
            ),
            # Forward lean - introduce after arms are stable
            "forward_lean": RewardSchedule(
                start_weight=0.0,
                end_weight=0.2,  # Match RewardsJoggingCfg target
                start_iteration=arm_start + 150,
                end_iteration=arm_end + 150,
                interpolation="cosine",
            ),
            # Joint default penalty - start low, increase
            "joint_default": RewardSchedule(
                start_weight=-0.02,  # Keep original low value initially
                end_weight=-0.08,
                start_iteration=arm_start,
                end_iteration=arm_end,
                interpolation="linear",
            ),
        },
        verbose=True,
    )


def create_conservative_scheduler(
    locomotion_warmup: int = 500,
    arm_transition_duration: int = 500,
) -> RewardScheduler:
    """Create a more conservative scheduler with slower transitions.

    Use this if the standard scheduler still causes instability.
    """
    arm_start = locomotion_warmup
    arm_end = locomotion_warmup + arm_transition_duration

    return RewardScheduler(
        schedules={
            "arm_close_to_body": RewardSchedule(
                start_weight=0.0,
                end_weight=0.15,  # Lower final weight
                start_iteration=arm_start,
                end_iteration=arm_end,
                interpolation="cosine",
            ),
            "arm_lateral_penalty": RewardSchedule(
                start_weight=0.0,
                end_weight=-0.3,  # Lower final weight
                start_iteration=arm_start,
                end_iteration=arm_end,
                interpolation="cosine",
            ),
            "elbow_bend": RewardSchedule(
                start_weight=0.0,
                end_weight=0.1,
                start_iteration=arm_start + 200,
                end_iteration=arm_end + 200,
                interpolation="cosine",
            ),
            "joint_default": RewardSchedule(
                start_weight=-0.02,
                end_weight=-0.05,
                start_iteration=arm_start,
                end_iteration=arm_end,
                interpolation="linear",
            ),
        },
        verbose=True,
    )
