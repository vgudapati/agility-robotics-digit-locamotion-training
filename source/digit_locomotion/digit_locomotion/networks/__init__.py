"""Neural network modules for Digit locomotion.

This module contains custom neural network architectures for locomotion control,
following the approach from "Real-world humanoid locomotion with reinforcement learning"
(Radosavovic et al., Science Robotics 2024).
"""

from .transformer_policy import ActorCriticTransformer, CausalTransformer

__all__ = ["ActorCriticTransformer", "CausalTransformer"]
