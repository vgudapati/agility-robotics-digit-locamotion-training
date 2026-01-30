"""Custom observation functions for Digit locomotion.

These observations are designed for proprioceptive locomotion control,
providing the policy with information about the robot's state without
requiring external sensors like cameras or LIDAR.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feet_positions_in_base_frame(env: ManagerBasedRLEnv, asset_cfg) -> torch.Tensor:
    """Get the positions of the feet relative to the robot base frame.

    This is useful for the policy to understand where its feet are
    relative to its body, which helps with balance and gait planning.

    Args:
        env: The environment instance.
        asset_cfg: Configuration for the robot asset.

    Returns:
        Tensor of shape (num_envs, num_feet * 3) containing foot positions.
    """
    asset = env.scene[asset_cfg.name]

    # Get base frame information
    base_pos = asset.data.root_pos_w
    base_quat = asset.data.root_quat_w

    # Get foot body indices (assuming toe bodies are the feet)
    foot_body_ids = asset_cfg.body_ids
    foot_pos_w = asset.data.body_pos_w[:, foot_body_ids, :]

    # Transform to base frame
    # Inverse rotation of base quaternion
    base_quat_inv = quat_conjugate(base_quat)

    # Position relative to base
    foot_pos_rel = foot_pos_w - base_pos.unsqueeze(1)

    # Rotate to base frame
    foot_pos_b = quat_rotate(base_quat_inv.unsqueeze(1).expand_as(foot_pos_rel[..., :1].expand(-1, -1, 4)), foot_pos_rel)

    return foot_pos_b.reshape(env.num_envs, -1)


def feet_velocities_in_base_frame(env: ManagerBasedRLEnv, asset_cfg) -> torch.Tensor:
    """Get the velocities of the feet relative to the robot base frame.

    Args:
        env: The environment instance.
        asset_cfg: Configuration for the robot asset.

    Returns:
        Tensor of shape (num_envs, num_feet * 3) containing foot velocities.
    """
    asset = env.scene[asset_cfg.name]

    # Get base frame information
    base_quat = asset.data.root_quat_w
    base_lin_vel = asset.data.root_lin_vel_w
    base_ang_vel = asset.data.root_ang_vel_w

    # Get foot body indices
    foot_body_ids = asset_cfg.body_ids
    foot_vel_w = asset.data.body_lin_vel_w[:, foot_body_ids, :]

    # Velocity relative to base (accounting for base motion)
    foot_vel_rel = foot_vel_w - base_lin_vel.unsqueeze(1)

    # Rotate to base frame
    base_quat_inv = quat_conjugate(base_quat)
    foot_vel_b = quat_rotate(base_quat_inv.unsqueeze(1).expand(-1, foot_vel_rel.shape[1], -1), foot_vel_rel)

    return foot_vel_b.reshape(env.num_envs, -1)


def gait_phase(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Get the current gait phase as a sinusoidal encoding.

    This provides the policy with a notion of timing for periodic gaits.
    The phase is encoded as (sin(phase), cos(phase)) for smooth interpolation.

    Args:
        env: The environment instance.

    Returns:
        Tensor of shape (num_envs, 2) containing sin and cos of gait phase.
    """
    # Use episode time to compute phase
    # Assumes a gait period of approximately 0.5 seconds (2 Hz walking frequency)
    gait_frequency = 2.0  # Hz
    phase = 2.0 * torch.pi * gait_frequency * env.episode_length_buf * env.step_dt

    return torch.stack([torch.sin(phase), torch.cos(phase)], dim=-1)


def height_scan(env: ManagerBasedRLEnv, sensor_cfg) -> torch.Tensor:
    """Get height measurements from a height scanner sensor.

    This provides the policy with local terrain information around the robot.

    Args:
        env: The environment instance.
        sensor_cfg: Configuration for the height scanner sensor.

    Returns:
        Tensor containing height measurements relative to base height.
    """
    sensor = env.scene[sensor_cfg.name]
    return sensor.data.ray_hits_w[..., 2] - env.scene["robot"].data.root_pos_w[:, 2:3]


# Helper functions for quaternion operations
def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    """Compute the conjugate of a quaternion (wxyz format)."""
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate a vector by a quaternion (wxyz format)."""
    q_w = q[..., 0:1]
    q_vec = q[..., 1:4]

    # v' = v + 2 * q_w * (q_vec x v) + 2 * (q_vec x (q_vec x v))
    t = 2.0 * torch.cross(q_vec, v, dim=-1)
    return v + q_w * t + torch.cross(q_vec, t, dim=-1)
