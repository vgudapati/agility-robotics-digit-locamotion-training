"""Domain randomization events for Digit locomotion training.

Domain randomization is crucial for sim-to-real transfer. By training
with varied simulation parameters, the policy learns to be robust to
the differences between simulation and the real world.

Events are organized by when they occur:
- startup: Once at the beginning of training
- reset: Each time an environment resets
- interval: Periodically during episodes
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg


def randomize_joint_stiffness_and_damping(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg,
    stiffness_range: tuple[float, float] = (0.8, 1.2),
    damping_range: tuple[float, float] = (0.8, 1.2),
) -> None:
    """Randomize joint PD gains to simulate actuator variation.

    Real actuators have manufacturing tolerances and wear that
    cause their behavior to differ from nominal specifications.

    Args:
        env: The environment instance.
        env_ids: Environment indices to randomize.
        asset_cfg: Configuration for the robot asset.
        stiffness_range: Multiplier range for stiffness (e.g., 0.8-1.2 = ±20%).
        damping_range: Multiplier range for damping.
    """
    asset = env.scene[asset_cfg.name]

    # Get number of joints
    num_joints = asset.num_joints
    num_envs = len(env_ids)

    # Sample random multipliers
    stiffness_mult = torch.empty(num_envs, num_joints, device=env.device).uniform_(*stiffness_range)
    damping_mult = torch.empty(num_envs, num_joints, device=env.device).uniform_(*damping_range)

    # Apply to actuator parameters
    # Note: This assumes the asset has accessible stiffness/damping parameters
    # The exact implementation depends on the actuator model being used
    for actuator_name, actuator in asset.actuators.items():
        if hasattr(actuator, "stiffness"):
            default_stiffness = actuator.stiffness.clone()
            actuator.stiffness[env_ids] = default_stiffness[env_ids] * stiffness_mult
        if hasattr(actuator, "damping"):
            default_damping = actuator.damping.clone()
            actuator.damping[env_ids] = default_damping[env_ids] * damping_mult


def randomize_joint_friction(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg,
    friction_range: tuple[float, float] = (0.5, 2.0),
) -> None:
    """Randomize joint friction to simulate mechanical variation.

    Joint friction varies due to lubrication, wear, and temperature.

    Args:
        env: The environment instance.
        env_ids: Environment indices to randomize.
        asset_cfg: Configuration for the robot asset.
        friction_range: Range of friction values.
    """
    asset = env.scene[asset_cfg.name]

    num_joints = asset.num_joints
    num_envs = len(env_ids)

    # Sample random friction values
    friction = torch.empty(num_envs, num_joints, device=env.device).uniform_(*friction_range)

    # Apply to joint properties
    # Note: Implementation depends on physics engine API
    asset.root_physx_view.set_dof_armatures(friction, indices=env_ids)


def add_base_mass(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg,
    mass_range: tuple[float, float] = (-2.0, 5.0),
) -> None:
    """Add random mass to the robot base to simulate payload variation.

    The robot may carry varying loads or have attachments that
    change its effective mass.

    Args:
        env: The environment instance.
        env_ids: Environment indices to randomize.
        asset_cfg: Configuration for the robot asset.
        mass_range: Range of additional mass in kg (can be negative).
    """
    asset = env.scene[asset_cfg.name]

    num_envs = len(env_ids)

    # Sample random additional mass
    added_mass = torch.empty(num_envs, device=env.device).uniform_(*mass_range)

    # Get current base mass and add randomization
    # Note: Exact API depends on Isaac Sim version
    base_body_idx = 0  # Typically the first body is the base
    current_mass = asset.root_physx_view.get_masses()

    new_mass = current_mass.clone()
    new_mass[env_ids, base_body_idx] += added_mass

    # Clamp to ensure positive mass
    new_mass = torch.clamp(new_mass, min=0.1)

    asset.root_physx_view.set_masses(new_mass, indices=env_ids)


def randomize_com_position(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg,
    com_offset_range: tuple[float, float, float] = (0.05, 0.05, 0.05),
) -> None:
    """Randomize the center of mass position of the base.

    Manufacturing tolerances and attachments can shift the CoM.

    Args:
        env: The environment instance.
        env_ids: Environment indices to randomize.
        asset_cfg: Configuration for the robot asset.
        com_offset_range: Maximum offset in (x, y, z) directions in meters.
    """
    asset = env.scene[asset_cfg.name]

    num_envs = len(env_ids)

    # Sample random CoM offsets
    com_offset = torch.zeros(num_envs, 3, device=env.device)
    for i, max_offset in enumerate(com_offset_range):
        com_offset[:, i] = torch.empty(num_envs, device=env.device).uniform_(-max_offset, max_offset)

    # Apply to base body
    # Note: Exact API depends on Isaac Sim version
    current_com = asset.root_physx_view.get_coms()
    new_com = current_com.clone()
    base_body_idx = 0
    new_com[env_ids, base_body_idx, :3] += com_offset

    asset.root_physx_view.set_coms(new_com, indices=env_ids)


def push_robot(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    velocity_range: dict[str, tuple[float, float]],
) -> None:
    """Apply random velocity perturbation to the robot base.

    Simulates external disturbances like bumps, wind, or collisions.

    Args:
        env: The environment instance.
        env_ids: Environment indices to push.
        velocity_range: Dict with keys 'x', 'y' and ranges for velocity in m/s.
    """
    robot = env.scene["robot"]

    num_envs = len(env_ids)

    # Get current velocities
    current_vel = robot.data.root_lin_vel_w.clone()

    # Add random velocity impulse
    if "x" in velocity_range:
        current_vel[env_ids, 0] += torch.empty(num_envs, device=env.device).uniform_(*velocity_range["x"])
    if "y" in velocity_range:
        current_vel[env_ids, 1] += torch.empty(num_envs, device=env.device).uniform_(*velocity_range["y"])
    if "z" in velocity_range:
        current_vel[env_ids, 2] += torch.empty(num_envs, device=env.device).uniform_(*velocity_range["z"])

    # Apply new velocities
    robot.write_root_velocity_to_sim(current_vel, env_ids=env_ids)


def randomize_motor_strength(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg,
    strength_range: tuple[float, float] = (0.8, 1.2),
) -> None:
    """Randomize motor torque limits to simulate actuator variation.

    Motors have varying peak torque capabilities due to manufacturing
    tolerances, temperature, and wear.

    Args:
        env: The environment instance.
        env_ids: Environment indices to randomize.
        asset_cfg: Configuration for the robot asset.
        strength_range: Multiplier range for motor strength.
    """
    asset = env.scene[asset_cfg.name]

    num_joints = asset.num_joints
    num_envs = len(env_ids)

    # Sample random strength multipliers
    strength_mult = torch.empty(num_envs, num_joints, device=env.device).uniform_(*strength_range)

    # Apply to effort limits
    for actuator_name, actuator in asset.actuators.items():
        if hasattr(actuator, "effort_limit"):
            default_limit = actuator.effort_limit.clone()
            actuator.effort_limit[env_ids] = default_limit[env_ids] * strength_mult


def randomize_action_delay(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    delay_range: tuple[int, int] = (0, 2),
) -> None:
    """Set random action delay for specified environments.

    Communication delays between the controller and robot cause
    actions to be applied with some latency.

    Args:
        env: The environment instance.
        env_ids: Environment indices to randomize.
        delay_range: Range of delay in simulation steps.
    """
    num_envs = len(env_ids)

    # Initialize delay buffer if needed
    if not hasattr(env, "_action_delay"):
        env._action_delay = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        env._action_history = []

    # Sample random delays
    delays = torch.randint(delay_range[0], delay_range[1] + 1, (num_envs,), device=env.device)
    env._action_delay[env_ids] = delays


def apply_observation_noise(
    env: ManagerBasedRLEnv,
    noise_level: float = 0.05,
) -> torch.Tensor:
    """Add Gaussian noise to observations.

    Sensor noise is ubiquitous in real systems. Training with noisy
    observations helps the policy be robust to this.

    Args:
        env: The environment instance.
        noise_level: Standard deviation of noise as fraction of observation range.

    Returns:
        Noisy observation tensor.
    """
    obs = env.obs_buf.clone()
    noise = torch.randn_like(obs) * noise_level
    return obs + noise
