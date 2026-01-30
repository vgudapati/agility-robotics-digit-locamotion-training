"""Termination conditions for Digit locomotion training.

Terminations define when an episode should end. There are two types:
1. Time-based termination (episode timeout)
2. Failure terminations (robot fell, bad state, etc.)

Proper termination conditions are important for:
- Safety (prevent unrealistic states)
- Training efficiency (don't waste time on failed episodes)
- Learning signal (large penalty on termination)
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg


def bad_orientation(
    env: ManagerBasedRLEnv,
    limit_angle: float = 0.5,
) -> torch.Tensor:
    """Terminate when the robot orientation exceeds the limit.

    Uses the projected gravity vector to detect excessive tilt.
    When the robot tilts too much, it's likely falling and the
    episode should end.

    Args:
        env: The environment instance.
        limit_angle: Maximum allowed tilt angle in radians (~30 degrees).

    Returns:
        Boolean tensor indicating which environments should terminate.
    """
    robot = env.scene["robot"]

    # Get projected gravity in body frame
    # When upright, gravity should be [0, 0, -1] in body frame
    projected_gravity = robot.data.projected_gravity_b

    # Compute tilt from xy components of gravity
    # sin(tilt_angle) ≈ sqrt(gx^2 + gy^2) for small angles
    tilt = torch.sqrt(
        projected_gravity[:, 0] ** 2 + projected_gravity[:, 1] ** 2
    )

    return tilt > torch.sin(torch.tensor(limit_angle))


def base_height_below_threshold(
    env: ManagerBasedRLEnv,
    minimum_height: float = 0.5,
    asset_cfg: SceneEntityCfg = None,
) -> torch.Tensor:
    """Terminate when the robot base drops below a minimum height.

    If the robot's pelvis/base is too low, it has likely fallen
    or is in an irrecoverable state.

    Args:
        env: The environment instance.
        minimum_height: Minimum allowed base height in meters.
        asset_cfg: Configuration for the robot asset.

    Returns:
        Boolean tensor indicating which environments should terminate.
    """
    if asset_cfg is not None:
        asset = env.scene[asset_cfg.name]
    else:
        asset = env.scene["robot"]

    base_height = asset.data.root_pos_w[:, 2]
    return base_height < minimum_height


def illegal_contact(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Terminate when illegal body parts make contact with the ground.

    For bipedal robots, contact on the torso or pelvis typically
    indicates a fall.

    Args:
        env: The environment instance.
        sensor_cfg: Configuration specifying which bodies to check.
        threshold: Force threshold to consider as contact (N).

    Returns:
        Boolean tensor indicating which environments should terminate.
    """
    contact_sensor = env.scene[sensor_cfg.name]

    # Get contact forces for specified bodies
    contact_forces = contact_sensor.data.net_forces_w_history[:, 0]

    # Check if any contact exceeds threshold
    contact_magnitudes = torch.norm(contact_forces, dim=-1)
    has_illegal_contact = torch.any(contact_magnitudes > threshold, dim=1)

    return has_illegal_contact


def joint_velocity_out_of_bounds(
    env: ManagerBasedRLEnv,
    max_velocity: float = 50.0,
) -> torch.Tensor:
    """Terminate when joint velocities exceed safe limits.

    Extremely high joint velocities indicate simulation instability
    or an unrealistic state.

    Args:
        env: The environment instance.
        max_velocity: Maximum allowed joint velocity in rad/s.

    Returns:
        Boolean tensor indicating which environments should terminate.
    """
    robot = env.scene["robot"]
    joint_vel = robot.data.joint_vel

    # Check if any joint exceeds the velocity limit
    max_vel_per_env = torch.max(torch.abs(joint_vel), dim=1)[0]
    return max_vel_per_env > max_velocity


def joint_position_out_of_bounds(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Terminate when joints exceed their position limits.

    This shouldn't happen with properly configured soft limits,
    but serves as a safety check.

    Args:
        env: The environment instance.

    Returns:
        Boolean tensor indicating which environments should terminate.
    """
    robot = env.scene["robot"]

    joint_pos = robot.data.joint_pos
    joint_limits = robot.data.soft_joint_pos_limits

    lower_limits = joint_limits[:, :, 0]
    upper_limits = joint_limits[:, :, 1]

    # Check if any joint is outside limits
    below_lower = torch.any(joint_pos < lower_limits, dim=1)
    above_upper = torch.any(joint_pos > upper_limits, dim=1)

    return below_lower | above_upper


def base_linear_velocity_out_of_bounds(
    env: ManagerBasedRLEnv,
    max_velocity: float = 10.0,
) -> torch.Tensor:
    """Terminate when base linear velocity is unrealistically high.

    Extremely high velocities indicate simulation instability.

    Args:
        env: The environment instance.
        max_velocity: Maximum allowed base velocity in m/s.

    Returns:
        Boolean tensor indicating which environments should terminate.
    """
    robot = env.scene["robot"]
    lin_vel = robot.data.root_lin_vel_w

    vel_magnitude = torch.norm(lin_vel, dim=1)
    return vel_magnitude > max_velocity


def base_angular_velocity_out_of_bounds(
    env: ManagerBasedRLEnv,
    max_velocity: float = 20.0,
) -> torch.Tensor:
    """Terminate when base angular velocity is unrealistically high.

    Args:
        env: The environment instance.
        max_velocity: Maximum allowed angular velocity in rad/s.

    Returns:
        Boolean tensor indicating which environments should terminate.
    """
    robot = env.scene["robot"]
    ang_vel = robot.data.root_ang_vel_w

    vel_magnitude = torch.norm(ang_vel, dim=1)
    return vel_magnitude > max_velocity
