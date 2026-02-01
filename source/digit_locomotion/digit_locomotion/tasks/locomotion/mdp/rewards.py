"""Custom reward functions for Digit bipedal locomotion.

The reward function is organized into four categories:
1. Command Tracking - Primary objectives (velocity tracking)
2. Stability - Keep the robot upright and balanced
3. Gait Quality - Encourage natural walking patterns
4. Regularization - Smooth, energy-efficient motion

Reference: These rewards are based on proven designs from bipedal locomotion
research, including work from ETH RSL, MIT, and Berkeley.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg


# =============================================================================
# COMMAND TRACKING REWARDS
# =============================================================================

def track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float = 0.25
) -> torch.Tensor:
    """Exponential reward for tracking commanded XY linear velocity.

    Uses exp(-error^2 / std^2) formulation which:
    - Provides smooth gradients near the target
    - Saturates gracefully for large errors
    - Allows tuning sensitivity via std parameter

    Args:
        env: The environment instance.
        command_name: Name of the velocity command in the command manager.
        std: Standard deviation for the exponential (lower = more strict).

    Returns:
        Reward tensor of shape (num_envs,).
    """
    # Get actual velocity in body frame
    lin_vel_b = env.scene["robot"].data.root_lin_vel_b[:, :2]

    # Get commanded velocity
    lin_vel_cmd = env.command_manager.get_command(command_name)[:, :2]

    # Compute squared error
    error_sq = torch.sum(torch.square(lin_vel_cmd - lin_vel_b), dim=1)

    return torch.exp(-error_sq / (std ** 2))


def track_ang_vel_z_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float = 0.25
) -> torch.Tensor:
    """Exponential reward for tracking commanded yaw angular velocity.

    Args:
        env: The environment instance.
        command_name: Name of the velocity command in the command manager.
        std: Standard deviation for the exponential.

    Returns:
        Reward tensor of shape (num_envs,).
    """
    # Get actual angular velocity around z-axis in body frame
    ang_vel_z = env.scene["robot"].data.root_ang_vel_b[:, 2]

    # Get commanded angular velocity
    ang_vel_cmd = env.command_manager.get_command(command_name)[:, 2]

    # Compute squared error
    error_sq = torch.square(ang_vel_cmd - ang_vel_z)

    return torch.exp(-error_sq / (std ** 2))


# =============================================================================
# STABILITY REWARDS (PENALTIES)
# =============================================================================

def lin_vel_z_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize vertical velocity to prevent bouncing and jumping.

    For stable walking, vertical velocity should be minimal.
    Large vertical velocities indicate the robot is bouncing or falling.

    Returns:
        Squared vertical velocity (positive value, to be used with negative weight).
    """
    return torch.square(env.scene["robot"].data.root_lin_vel_b[:, 2])


def ang_vel_xy_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize roll and pitch angular velocities for smooth body motion.

    Excessive rolling or pitching indicates unstable walking.

    Returns:
        Sum of squared roll and pitch angular velocities.
    """
    ang_vel_xy = env.scene["robot"].data.root_ang_vel_b[:, :2]
    return torch.sum(torch.square(ang_vel_xy), dim=1)


def flat_orientation_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize deviation from flat (upright) orientation.

    Uses the projected gravity vector in body frame. When the robot is
    upright, gravity should point straight down in body frame [0, 0, -1].
    Deviation in x/y components indicates tilt.

    Returns:
        Sum of squared x and y components of projected gravity.
    """
    projected_gravity = env.scene["robot"].data.projected_gravity_b[:, :2]
    return torch.sum(torch.square(projected_gravity), dim=1)


def base_height_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Penalize deviation from target base height.

    Helps maintain consistent standing height during locomotion.

    Args:
        env: The environment instance.
        target_height: Desired height of the robot base in meters.
        asset_cfg: Configuration for the robot asset.

    Returns:
        Squared deviation from target height.
    """
    asset = env.scene[asset_cfg.name]
    base_height = asset.data.root_pos_w[:, 2]
    return torch.square(base_height - target_height)


# =============================================================================
# GAIT QUALITY REWARDS
# =============================================================================

def feet_air_time(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    threshold: float = 0.5,
    target_air_time: float = 0.3,
) -> torch.Tensor:
    """Reward feet spending appropriate time in the air during walking.

    This encourages a proper stepping gait rather than shuffling.
    The reward is given when a foot makes contact after being in the air.

    Args:
        env: The environment instance.
        sensor_cfg: Configuration for the contact force sensor.
        command_name: Name of the velocity command.
        threshold: Force threshold to detect contact (N).
        target_air_time: Target duration for foot to be in air (seconds).

    Returns:
        Air time reward.
    """
    # Get contact forces
    contact_sensor = env.scene[sensor_cfg.name]
    contact_forces = contact_sensor.data.net_forces_w_history[:, 0]  # Current timestep

    # Detect contact (force above threshold)
    contact = torch.norm(contact_forces, dim=-1) > threshold

    # Get previous contact state from environment
    # Note: This requires the environment to track _prev_feet_contact
    if not hasattr(env, "_prev_feet_contact"):
        env._prev_feet_contact = torch.zeros_like(contact)
        env._feet_air_time = torch.zeros(env.num_envs, contact.shape[1], device=env.device)

    # Detect first contact (transition from air to ground)
    first_contact = contact & ~env._prev_feet_contact

    # Update air time counter
    env._feet_air_time += env.step_dt

    # Compute reward based on air time when foot makes contact
    # Reward is proportional to how close air time is to target
    air_time_reward = torch.sum(
        (env._feet_air_time - target_air_time) * first_contact.float(),
        dim=1
    )

    # Reset air time counter on contact
    env._feet_air_time = env._feet_air_time * (~contact).float()

    # Update previous contact state
    env._prev_feet_contact = contact.clone()

    # Only reward when robot should be moving
    cmd_vel = env.command_manager.get_command(command_name)
    moving = torch.norm(cmd_vel[:, :2], dim=1) > 0.1

    return air_time_reward * moving.float()


def foot_clearance_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target_clearance: float = 0.06,
) -> torch.Tensor:
    """Reward feet reaching target height during swing phase.

    This prevents foot dragging and encourages proper stepping.

    Args:
        env: The environment instance.
        asset_cfg: Configuration specifying foot body names.
        target_clearance: Target foot height during swing (meters).

    Returns:
        Foot clearance reward.
    """
    asset = env.scene[asset_cfg.name]

    # Get foot heights
    foot_body_ids = asset_cfg.body_ids
    foot_heights = asset.data.body_pos_w[:, foot_body_ids, 2]

    # Get terrain height (assume flat terrain at z=0 for simplicity)
    terrain_height = 0.0

    # Compute clearance
    clearance = foot_heights - terrain_height

    # Reward for achieving target clearance
    # Use a soft threshold to avoid discontinuities
    reward = torch.sum(torch.clamp(clearance - target_clearance, min=-0.1, max=0.1), dim=1)

    return reward


def gait_symmetry(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward symmetric gait between left and right legs.

    Encourages natural bipedal walking pattern where left and right
    legs alternate in a symmetric fashion.

    Returns:
        Symmetry reward (higher is more symmetric).
    """
    robot = env.scene["robot"]

    # Get joint positions for left and right legs
    # Assumes joint ordering: [left_leg_joints..., right_leg_joints...]
    joint_pos = robot.data.joint_pos

    # Number of joints per leg (typically 8 for Digit)
    n_leg_joints = 8

    left_leg_pos = joint_pos[:, :n_leg_joints]
    right_leg_pos = joint_pos[:, n_leg_joints:2*n_leg_joints]

    # Compute asymmetry as difference between mirrored positions
    # For walking, left at phase phi should equal right at phase phi + pi
    # Simplified: penalize large differences when robot is moving
    asymmetry = torch.sum(torch.abs(left_leg_pos - right_leg_pos), dim=1)

    return -asymmetry


# =============================================================================
# REGULARIZATION PENALTIES
# =============================================================================

def action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize rapid changes in actions for smoother motion.

    Large action changes can cause jerky motion and stress actuators.

    Returns:
        Squared difference between current and previous actions.
    """
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)


def joint_acc_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize joint accelerations for smoother motion.

    Returns:
        Sum of squared joint accelerations.
    """
    robot = env.scene["robot"]
    return torch.sum(torch.square(robot.data.joint_acc), dim=1)


def joint_torques_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize joint torques for energy efficiency.

    Lower torques generally mean more efficient motion and
    less stress on actuators.

    Returns:
        Sum of squared applied torques.
    """
    robot = env.scene["robot"]
    return torch.sum(torch.square(robot.data.applied_torque), dim=1)


def joint_vel_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize high joint velocities.

    Helps prevent overly aggressive motions.

    Returns:
        Sum of squared joint velocities.
    """
    robot = env.scene["robot"]
    return torch.sum(torch.square(robot.data.joint_vel), dim=1)


def joint_pos_limits(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joints approaching their position limits.

    Encourages the robot to stay away from joint limits where
    motion becomes constrained.

    Args:
        env: The environment instance.
        asset_cfg: Configuration for the robot asset.

    Returns:
        Penalty for joints near limits.
    """
    asset = env.scene[asset_cfg.name]

    # Get current joint positions
    joint_pos = asset.data.joint_pos

    # Get joint limits
    joint_pos_limits = asset.data.soft_joint_pos_limits
    lower_limits = joint_pos_limits[:, :, 0]
    upper_limits = joint_pos_limits[:, :, 1]

    # Compute distance to limits
    dist_to_lower = joint_pos - lower_limits
    dist_to_upper = upper_limits - joint_pos

    # Penalize being within 10% of the limit range
    limit_range = upper_limits - lower_limits
    threshold = 0.1 * limit_range

    lower_penalty = torch.sum(torch.clamp(threshold - dist_to_lower, min=0.0), dim=1)
    upper_penalty = torch.sum(torch.clamp(threshold - dist_to_upper, min=0.0), dim=1)

    return lower_penalty + upper_penalty


def undesired_contacts(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Penalize contacts on body parts that shouldn't touch the ground.

    For Digit, we want to penalize contacts on the torso, thighs, etc.
    Only the feet should make contact during normal walking.

    Args:
        env: The environment instance.
        sensor_cfg: Configuration specifying which bodies to check.
        threshold: Force threshold to consider as contact (N).

    Returns:
        Penalty for undesired contacts.
    """
    contact_sensor = env.scene[sensor_cfg.name]
    contact_forces = contact_sensor.data.net_forces_w_history[:, 0]

    # Count bodies with contact above threshold
    contact_magnitudes = torch.norm(contact_forces, dim=-1)
    num_contacts = torch.sum((contact_magnitudes > threshold).float(), dim=1)

    return num_contacts


def stand_still_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float = 0.1,
) -> torch.Tensor:
    """Penalize motion when commanded to stand still.

    When velocity command is near zero, the robot should minimize motion.

    Args:
        env: The environment instance.
        command_name: Name of the velocity command.
        threshold: Velocity threshold below which standing is expected.

    Returns:
        Penalty for moving when should be standing.
    """
    # Get commanded velocity magnitude
    cmd_vel = env.command_manager.get_command(command_name)
    cmd_vel_magnitude = torch.norm(cmd_vel[:, :2], dim=1)

    # Check if should be standing
    should_stand = cmd_vel_magnitude < threshold

    # Penalize joint velocities when standing
    robot = env.scene["robot"]
    motion_penalty = torch.sum(torch.square(robot.data.joint_vel), dim=1)

    return motion_penalty * should_stand.float()


# =============================================================================
# ARM SWING COORDINATION
# =============================================================================

def arm_swing_coordination(
    env: ManagerBasedRLEnv,
    command_name: str,
) -> torch.Tensor:
    """Reward natural arm swing in opposition to legs (like human walking/running).

    When walking/running, humans swing their arms opposite to their legs:
    - Left leg forward -> Right arm forward
    - Right leg forward -> Left arm forward

    This creates counter-rotation that helps balance and is more energy efficient.

    The reward is computed by checking if:
    - left_hip_pitch velocity * right_shoulder_pitch velocity > 0 (same direction)
    - right_hip_pitch velocity * left_shoulder_pitch velocity > 0 (same direction)

    Joint names for Digit V4:
    - Leg pitch: left_hip_pitch, right_hip_pitch
    - Arm pitch: left_shoulder_pitch, right_shoulder_pitch

    Args:
        env: The environment instance.
        command_name: Name of the velocity command (to disable when standing).

    Returns:
        Reward for coordinated arm swing (positive when properly coordinated).
    """
    robot = env.scene["robot"]

    # Get joint velocities
    joint_vel = robot.data.joint_vel
    joint_names = robot.data.joint_names

    # Find joint indices for hip and shoulder pitch joints
    # These control the forward/backward swing motion
    left_hip_pitch_idx = None
    right_hip_pitch_idx = None
    left_shoulder_pitch_idx = None
    right_shoulder_pitch_idx = None

    for i, name in enumerate(joint_names):
        if "left_hip_pitch" in name:
            left_hip_pitch_idx = i
        elif "right_hip_pitch" in name:
            right_hip_pitch_idx = i
        elif "left_shoulder_pitch" in name:
            left_shoulder_pitch_idx = i
        elif "right_shoulder_pitch" in name:
            right_shoulder_pitch_idx = i

    # If joints not found, return zero reward
    if any(idx is None for idx in [left_hip_pitch_idx, right_hip_pitch_idx,
                                    left_shoulder_pitch_idx, right_shoulder_pitch_idx]):
        return torch.zeros(env.num_envs, device=env.device)

    # Get velocities for the relevant joints
    left_hip_vel = joint_vel[:, left_hip_pitch_idx]
    right_hip_vel = joint_vel[:, right_hip_pitch_idx]
    left_shoulder_vel = joint_vel[:, left_shoulder_pitch_idx]
    right_shoulder_vel = joint_vel[:, right_shoulder_pitch_idx]

    # Compute coordination reward:
    # - Left hip and right shoulder should move in SAME direction
    # - Right hip and left shoulder should move in SAME direction
    # Using product of velocities: positive when same direction, negative when opposite
    coordination_left = left_hip_vel * right_shoulder_vel  # Should be positive
    coordination_right = right_hip_vel * left_shoulder_vel  # Should be positive

    # Reward is sum of coordination scores (positive = good coordination)
    # Use tanh to bound the reward and provide smooth gradients
    reward = torch.tanh(coordination_left) + torch.tanh(coordination_right)

    # Only reward when robot should be moving
    cmd_vel = env.command_manager.get_command(command_name)
    moving = torch.norm(cmd_vel[:, :2], dim=1) > 0.1

    return reward * moving.float()


def joint_default_position(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize joints deviating from their default positions.

    This helps keep arms in a natural position and prevents extreme poses.
    The penalty is the squared distance from default joint positions.

    Args:
        env: The environment instance.
        asset_cfg: Configuration for the robot asset.

    Returns:
        Squared deviation from default positions.
    """
    asset = env.scene[asset_cfg.name]

    # Get current joint positions
    joint_pos = asset.data.joint_pos

    # Get default joint positions
    default_joint_pos = asset.data.default_joint_pos

    # Compute squared deviation
    deviation = torch.sum(torch.square(joint_pos - default_joint_pos), dim=1)

    return deviation
