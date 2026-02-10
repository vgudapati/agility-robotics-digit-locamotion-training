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
# SURVIVAL REWARDS
# =============================================================================

def is_alive(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return 1.0 for all environments that are still alive.

    This serves as a survival bonus - the agent gets +1 reward for each
    timestep it stays alive. This encourages the agent to survive longer
    rather than dying quickly to minimize accumulated negative rewards.

    Returns:
        Tensor of 1.0 for all environments (since dead envs are reset).
    """
    return torch.ones(env.num_envs, device=env.device)


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

# Global flag to track if we've printed joint names debug info
_arm_swing_debug_printed = False

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
    - left_hip_pitch velocity * right_arm_pitch velocity > 0 (same direction)
    - right_hip_pitch velocity * left_arm_pitch velocity > 0 (same direction)

    Joint names for Digit V4 (Isaac Lab convention):
    - Leg pitch: left_leg_hip_pitch, right_leg_hip_pitch
    - Arm shoulder pitch: left_arm_shoulder_pitch, right_arm_shoulder_pitch
      (NOT wrist_pitch - we want the shoulder joint for arm swing)

    Args:
        env: The environment instance.
        command_name: Name of the velocity command (to disable when standing).

    Returns:
        Reward for coordinated arm swing (positive when properly coordinated).
    """
    global _arm_swing_debug_printed
    robot = env.scene["robot"]

    # Get joint velocities
    joint_vel = robot.data.joint_vel
    joint_names = robot.data.joint_names

    # Debug: Print joint names once at startup
    if not _arm_swing_debug_printed:
        print(f"[arm_swing DEBUG] Total joints: {len(joint_names)}")
        print(f"[arm_swing DEBUG] Joint names: {list(joint_names)}")
        _arm_swing_debug_printed = True

    # Find joint indices for hip pitch and arm pitch joints
    # These control the forward/backward swing motion
    # Digit V4 naming: .*_leg_hip_pitch, .*_arm_pitch
    left_hip_pitch_idx = None
    right_hip_pitch_idx = None
    left_arm_pitch_idx = None
    right_arm_pitch_idx = None

    for i, name in enumerate(joint_names):
        name_lower = name.lower()
        # Match hip pitch joints (leg forward/backward)
        # Patterns: left_hip_pitch, left_leg_hip_pitch, etc.
        if "left" in name_lower and "hip" in name_lower and "pitch" in name_lower:
            left_hip_pitch_idx = i
        elif "right" in name_lower and "hip" in name_lower and "pitch" in name_lower:
            right_hip_pitch_idx = i
        # Match arm SHOULDER pitch joints (arm forward/backward swing)
        # Must match "shoulder" specifically to avoid matching wrist_pitch
        # Digit V4 naming: left_arm_shoulder_pitch, right_arm_shoulder_pitch
        elif "left" in name_lower and "shoulder" in name_lower and "pitch" in name_lower:
            left_arm_pitch_idx = i
        elif "right" in name_lower and "shoulder" in name_lower and "pitch" in name_lower:
            right_arm_pitch_idx = i

    # If joints not found, return zero reward
    if any(idx is None for idx in [left_hip_pitch_idx, right_hip_pitch_idx,
                                    left_arm_pitch_idx, right_arm_pitch_idx]):
        # Debug: print which joints were not found
        print(f"[arm_swing DEBUG] Joint matching FAILED!")
        print(f"[arm_swing DEBUG] Found: hip_l={left_hip_pitch_idx}, hip_r={right_hip_pitch_idx}, "
              f"arm_l={left_arm_pitch_idx}, arm_r={right_arm_pitch_idx}")
        return torch.zeros(env.num_envs, device=env.device)
    else:
        # Debug: print success message once
        if not hasattr(arm_swing_coordination, '_success_printed'):
            print(f"[arm_swing DEBUG] Joint matching SUCCESS!")
            print(f"[arm_swing DEBUG] Indices: hip_l={left_hip_pitch_idx}, hip_r={right_hip_pitch_idx}, "
                  f"arm_l={left_arm_pitch_idx}, arm_r={right_arm_pitch_idx}")
            arm_swing_coordination._success_printed = True

    # Get velocities for the relevant joints
    left_hip_vel = joint_vel[:, left_hip_pitch_idx]
    right_hip_vel = joint_vel[:, right_hip_pitch_idx]
    left_arm_vel = joint_vel[:, left_arm_pitch_idx]
    right_arm_vel = joint_vel[:, right_arm_pitch_idx]

    # Compute coordination reward:
    # - Left hip and right arm should move in SAME direction
    # - Right hip and left arm should move in SAME direction
    # Using product of velocities: positive when same direction, negative when opposite
    coordination_left = left_hip_vel * right_arm_vel  # Should be positive
    coordination_right = right_hip_vel * left_arm_vel  # Should be positive

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


# =============================================================================
# JOGGING/RUNNING POSTURE REWARDS
# =============================================================================

def forward_lean_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    target_lean_per_speed: float = 0.05,  # radians per m/s
    max_lean: float = 0.2,  # ~11 degrees max
) -> torch.Tensor:
    """Reward slight forward lean proportional to forward speed.

    When jogging/running, humans naturally lean forward. The lean angle
    increases with speed for aerodynamic efficiency and momentum.

    Uses the projected gravity vector in body frame:
    - Upright: gravity = [0, 0, -1]
    - Forward lean: gravity_x becomes positive (gravity points slightly backward in body frame)

    Args:
        env: The environment instance.
        command_name: Name of the velocity command.
        target_lean_per_speed: Target lean angle per m/s of forward velocity.
        max_lean: Maximum lean angle in radians.

    Returns:
        Reward for appropriate forward lean (negative error from target).
    """
    robot = env.scene["robot"]

    # Get commanded forward velocity
    cmd_vel = env.command_manager.get_command(command_name)
    forward_vel = torch.clamp(cmd_vel[:, 0], min=0.0)  # Only positive (forward) velocity

    # Calculate target lean based on speed
    target_lean = torch.clamp(forward_vel * target_lean_per_speed, max=max_lean)

    # Get actual lean from projected gravity
    # When leaning forward, gravity_x in body frame becomes positive
    projected_gravity = robot.data.projected_gravity_b
    actual_lean = projected_gravity[:, 0]  # Positive = leaning forward

    # Reward for being close to target lean
    lean_error = torch.abs(actual_lean - target_lean)

    # Only apply when moving forward
    moving_forward = forward_vel > 0.5  # At least 0.5 m/s forward

    return -lean_error * moving_forward.float()


def elbow_bend_while_moving(
    env: ManagerBasedRLEnv,
    command_name: str,
    target_bend: float = 0.8,  # ~45 degrees
) -> torch.Tensor:
    """Reward bent elbows during locomotion (like human running).

    Humans bend their elbows when jogging/running for more efficient
    arm swing and reduced moment of inertia.

    Args:
        env: The environment instance.
        command_name: Name of the velocity command.
        target_bend: Target elbow bend angle in radians (~0.8 = 45 degrees).

    Returns:
        Negative distance from target bend (to be used with positive weight).
    """
    robot = env.scene["robot"]
    joint_pos = robot.data.joint_pos
    joint_names = robot.data.joint_names

    # Find elbow joint indices
    left_elbow_idx = None
    right_elbow_idx = None

    for i, name in enumerate(joint_names):
        name_lower = name.lower()
        if "left" in name_lower and "elbow" in name_lower:
            left_elbow_idx = i
        elif "right" in name_lower and "elbow" in name_lower:
            right_elbow_idx = i

    # If joints not found, return zero
    if left_elbow_idx is None or right_elbow_idx is None:
        return torch.zeros(env.num_envs, device=env.device)

    # Get elbow positions
    left_elbow = joint_pos[:, left_elbow_idx]
    right_elbow = joint_pos[:, right_elbow_idx]

    # Reward being close to target bend angle
    bend_error = torch.abs(left_elbow - target_bend) + torch.abs(right_elbow - target_bend)

    # Only when moving
    cmd_vel = env.command_manager.get_command(command_name)
    moving = torch.norm(cmd_vel[:, :2], dim=1) > 0.5  # At least 0.5 m/s

    return -bend_error * moving.float()


def arm_lateral_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    max_lateral: float = 0.3,  # ~17 degrees max lateral extension
) -> torch.Tensor:
    """Penalize arms extending laterally (sideways) during locomotion.

    Human arms should swing forward/backward during running, not out to the sides.
    The shoulder roll joint controls lateral arm movement.

    Args:
        env: The environment instance.
        command_name: Name of the velocity command.
        max_lateral: Maximum allowed lateral extension before penalty (radians).

    Returns:
        Penalty for lateral arm extension (positive value, use with negative weight).
    """
    robot = env.scene["robot"]
    joint_pos = robot.data.joint_pos
    joint_names = robot.data.joint_names

    # Find shoulder roll joint indices (controls lateral arm movement)
    left_shoulder_roll_idx = None
    right_shoulder_roll_idx = None

    for i, name in enumerate(joint_names):
        name_lower = name.lower()
        if "left" in name_lower and "shoulder" in name_lower and "roll" in name_lower:
            left_shoulder_roll_idx = i
        elif "right" in name_lower and "shoulder" in name_lower and "roll" in name_lower:
            right_shoulder_roll_idx = i

    # If joints not found, return zero
    if left_shoulder_roll_idx is None or right_shoulder_roll_idx is None:
        return torch.zeros(env.num_envs, device=env.device)

    # Get shoulder roll positions (how far arms are extended sideways)
    left_roll = joint_pos[:, left_shoulder_roll_idx]
    right_roll = joint_pos[:, right_shoulder_roll_idx]

    # Penalize lateral extension beyond threshold
    # Using soft penalty with squared error beyond threshold
    left_excess = torch.clamp(torch.abs(left_roll) - max_lateral, min=0.0)
    right_excess = torch.clamp(torch.abs(right_roll) - max_lateral, min=0.0)

    penalty = torch.square(left_excess) + torch.square(right_excess)

    # Only when moving
    cmd_vel = env.command_manager.get_command(command_name)
    moving = torch.norm(cmd_vel[:, :2], dim=1) > 0.3

    return penalty * moving.float()


def arm_close_to_body(
    env: ManagerBasedRLEnv,
    command_name: str,
) -> torch.Tensor:
    """Reward keeping arms close to the body during locomotion.

    Proper running form has arms swinging close to the torso,
    not flailing outward. This rewards minimal shoulder roll angles.

    Args:
        env: The environment instance.
        command_name: Name of the velocity command.

    Returns:
        Negative of arm deviation from body (use with positive weight).
    """
    robot = env.scene["robot"]
    joint_pos = robot.data.joint_pos
    joint_names = robot.data.joint_names

    # Find shoulder roll and yaw joint indices
    arm_joint_indices = []

    for i, name in enumerate(joint_names):
        name_lower = name.lower()
        # Match shoulder roll and yaw (lateral arm position control)
        if "shoulder" in name_lower and ("roll" in name_lower or "yaw" in name_lower):
            arm_joint_indices.append(i)

    # If joints not found, return zero
    if len(arm_joint_indices) == 0:
        return torch.zeros(env.num_envs, device=env.device)

    # Sum squared deviation from zero (neutral position)
    deviation = torch.sum(torch.square(joint_pos[:, arm_joint_indices]), dim=1)

    # Only when moving
    cmd_vel = env.command_manager.get_command(command_name)
    moving = torch.norm(cmd_vel[:, :2], dim=1) > 0.3

    return -deviation * moving.float()


def arm_leg_phase_coordination(
    env: ManagerBasedRLEnv,
    command_name: str,
) -> torch.Tensor:
    """Reward arm swing that opposes leg motion (position-based).

    Natural human locomotion has arms swinging in opposition to legs:
    - When left leg is forward (positive hip pitch), right arm should be forward
    - When right leg is forward, left arm should be forward

    This uses POSITION correlation rather than velocity to ensure arms
    reach equal amplitude forward and backward.

    The reward is computed as:
        correlation = left_hip_pos * right_shoulder_pitch + right_hip_pos * left_shoulder_pitch

    Higher correlation = better opposition (arms swing opposite to legs).

    Args:
        env: The environment instance.
        command_name: Name of the velocity command.

    Returns:
        Correlation reward (positive when arms oppose legs).
    """
    robot = env.scene["robot"]
    joint_pos = robot.data.joint_pos
    joint_names = robot.data.joint_names

    # Find joint indices
    left_hip_pitch_idx = None
    right_hip_pitch_idx = None
    left_shoulder_pitch_idx = None
    right_shoulder_pitch_idx = None

    for i, name in enumerate(joint_names):
        name_lower = name.lower()
        if "left" in name_lower and "hip" in name_lower and "pitch" in name_lower:
            left_hip_pitch_idx = i
        elif "right" in name_lower and "hip" in name_lower and "pitch" in name_lower:
            right_hip_pitch_idx = i
        elif "left" in name_lower and "shoulder" in name_lower and "pitch" in name_lower:
            left_shoulder_pitch_idx = i
        elif "right" in name_lower and "shoulder" in name_lower and "pitch" in name_lower:
            right_shoulder_pitch_idx = i

    # If joints not found, return zero
    if any(idx is None for idx in [left_hip_pitch_idx, right_hip_pitch_idx,
                                    left_shoulder_pitch_idx, right_shoulder_pitch_idx]):
        return torch.zeros(env.num_envs, device=env.device)

    # Get joint positions
    left_hip_pos = joint_pos[:, left_hip_pitch_idx]
    right_hip_pos = joint_pos[:, right_hip_pitch_idx]
    left_arm_pos = joint_pos[:, left_shoulder_pitch_idx]
    right_arm_pos = joint_pos[:, right_shoulder_pitch_idx]

    # Compute phase coordination:
    # Left hip forward (positive) should correlate with right arm forward
    # Right hip forward should correlate with left arm forward
    # Using product: positive when in phase, negative when out of phase
    coordination = left_hip_pos * right_arm_pos + right_hip_pos * left_arm_pos

    # Normalize with tanh for bounded gradients
    reward = torch.tanh(coordination)

    # Only when moving
    cmd_vel = env.command_manager.get_command(command_name)
    moving = torch.norm(cmd_vel[:, :2], dim=1) > 0.3

    return reward * moving.float()


def arm_swing_center_bias_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    neutral_pitch: float = 0.0,
) -> torch.Tensor:
    """Penalize arm swing bias (arms consistently forward or backward).

    During natural running, arm swing should be symmetric around the body -
    arms should reach equally forward and backward. If the MEAN shoulder pitch
    is consistently non-zero, it indicates a bias (e.g., arms always back).

    This penalty uses the MEAN of left and right shoulder pitch to detect
    systematic bias. Individual arm positions can vary, but the average
    should stay near neutral.

    Penalty = (mean_shoulder_pitch - neutral_pitch)^2

    Args:
        env: The environment instance.
        command_name: Name of the velocity command.
        neutral_pitch: Target neutral shoulder pitch angle (default 0.0).

    Returns:
        Squared deviation of mean arm position from neutral (use with negative weight).
    """
    robot = env.scene["robot"]
    joint_pos = robot.data.joint_pos
    joint_names = robot.data.joint_names

    # Find shoulder pitch indices
    left_shoulder_pitch_idx = None
    right_shoulder_pitch_idx = None

    for i, name in enumerate(joint_names):
        name_lower = name.lower()
        if "left" in name_lower and "shoulder" in name_lower and "pitch" in name_lower:
            left_shoulder_pitch_idx = i
        elif "right" in name_lower and "shoulder" in name_lower and "pitch" in name_lower:
            right_shoulder_pitch_idx = i

    # If joints not found, return zero
    if left_shoulder_pitch_idx is None or right_shoulder_pitch_idx is None:
        return torch.zeros(env.num_envs, device=env.device)

    # Get shoulder pitch positions
    left_arm_pitch = joint_pos[:, left_shoulder_pitch_idx]
    right_arm_pitch = joint_pos[:, right_shoulder_pitch_idx]

    # Compute mean shoulder pitch (bias indicator)
    mean_pitch = (left_arm_pitch + right_arm_pitch) / 2.0

    # Penalty for deviation from neutral
    bias_penalty = torch.square(mean_pitch - neutral_pitch)

    # Only when moving
    cmd_vel = env.command_manager.get_command(command_name)
    moving = torch.norm(cmd_vel[:, :2], dim=1) > 0.3

    return bias_penalty * moving.float()


def shoulder_roll_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize shoulder roll deviation from default position (arms-at-sides).

    The broad `joint_deviation_l1` on all arm joints dilutes the shoulder roll
    signal across pitch, yaw, elbow, and wrist joints. This reward targets
    ONLY shoulder roll joints with L2 penalty for a stronger gradient near
    the target (default pose = arms down at sides).

    Uses default_joint_pos as target, which for Digit V4 corresponds to arms
    hanging naturally at the robot's sides.

    Args:
        env: The environment instance.
        asset_cfg: Configuration specifying shoulder roll joints.

    Returns:
        Sum of squared shoulder roll deviation from default (use with negative weight).
    """
    asset = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    default_pos = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(joint_pos - default_pos), dim=1)


def joint_position_target_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target: float = 0.0,
) -> torch.Tensor:
    """Penalize joint deviation from a specified target angle (not default).

    Unlike shoulder_roll_penalty which targets default_joint_pos, this function
    targets an arbitrary angle. Useful for encouraging specific postures like
    elbow bend during running.

    Args:
        env: The environment instance.
        asset_cfg: Configuration specifying which joints to target.
        target: Target angle in radians for all specified joints.

    Returns:
        Sum of squared deviation from target (use with negative weight).
    """
    asset = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(joint_pos - target), dim=1)


def excessive_forward_lean_penalty(
    env: ManagerBasedRLEnv,
    max_lean: float = 0.1,  # ~6 degrees max forward lean
) -> torch.Tensor:
    """Penalize excessive forward body lean.

    While some forward lean is natural during running, excessive lean
    causes the arms to swing backward to compensate. This penalty keeps
    the body closer to vertical (90 degrees to ground).

    Uses projected gravity in body frame:
    - Upright: gravity = [0, 0, -1]
    - Forward lean: gravity_x becomes positive

    Penalty applies when lean exceeds max_lean threshold.

    Args:
        env: The environment instance.
        max_lean: Maximum allowed forward lean before penalty (radians).
                  Default 0.1 rad ≈ 6 degrees.

    Returns:
        Squared excess lean (use with negative weight).
    """
    robot = env.scene["robot"]

    # Get forward lean from projected gravity
    # When leaning forward, gravity_x in body frame becomes positive
    projected_gravity = robot.data.projected_gravity_b
    forward_lean = projected_gravity[:, 0]  # Positive = leaning forward

    # Only penalize forward lean that exceeds threshold
    excess_lean = torch.clamp(forward_lean - max_lean, min=0.0)

    return torch.square(excess_lean)


def mechanical_power_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize mechanical power consumption (torque * velocity).

    True energy cost is |torque × angular_velocity| summed over all joints.
    Unlike joint_torques_l2 (which only penalizes torque magnitude), this
    captures the actual work done by actuators — flailing arms at high velocity
    with moderate torque will be heavily penalized.

    Uses applied_torque (after actuator model) rather than commanded action,
    so it reflects real physical energy expenditure in simulation.

    Returns:
        Sum of |applied_torque * joint_vel| across all joints (use with negative weight).
    """
    robot = env.scene["robot"]
    return torch.sum(torch.abs(robot.data.applied_torque * robot.data.joint_vel), dim=1)


def upright_posture_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
) -> torch.Tensor:
    """Reward maintaining upright posture (body perpendicular to ground).

    Encourages the robot to stay close to 90 degrees relative to ground,
    regardless of speed. This prevents excessive forward lean that causes
    unnatural arm swing patterns.

    Uses exponential reward for smooth gradients near target.

    Args:
        env: The environment instance.
        command_name: Name of the velocity command.

    Returns:
        Exponential reward for upright posture (1.0 when perfectly upright).
    """
    robot = env.scene["robot"]

    # Get tilt from projected gravity
    # Perfect upright: gravity = [0, 0, -1], so x and y components = 0
    projected_gravity = robot.data.projected_gravity_b
    tilt_xy = projected_gravity[:, :2]

    # Compute squared tilt magnitude
    tilt_sq = torch.sum(torch.square(tilt_xy), dim=1)

    # Exponential reward: 1.0 when upright, decays with tilt
    # std=0.1 means ~60% reward at ~6 degrees tilt
    reward = torch.exp(-tilt_sq / (0.1 ** 2))

    # Only when moving (standing still has its own posture requirements)
    cmd_vel = env.command_manager.get_command(command_name)
    moving = torch.norm(cmd_vel[:, :2], dim=1) > 0.3

    return reward * moving.float()
