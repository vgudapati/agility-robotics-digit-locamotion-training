"""Environment configuration for Digit velocity-tracking locomotion.

This module defines the complete MDP configuration for training a
locomotion policy that tracks commanded velocities. The configuration
follows Isaac Lab's manager-based workflow.

Two environments are provided:
1. DigitFlatEnvCfg - Training on flat terrain (recommended for initial training)
2. DigitRoughEnvCfg - Training on rough/varied terrain (for robust policies)
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    SceneEntityCfg,
    TerminationTermCfg,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg, AdditiveGaussianNoiseCfg

# Import MDP components
import isaaclab.envs.mdp as mdp

# Import Digit robot configuration
from digit_locomotion.assets.digit import DIGIT_CFG


# =============================================================================
# SCENE CONFIGURATION
# =============================================================================

@configclass
class DigitSceneCfg(InteractiveSceneCfg):
    """Scene configuration for Digit locomotion environment."""

    # Ground plane
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )

    # Digit robot
    robot: ArticulationCfg = DIGIT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Contact sensors for feet
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
        update_period=0.0,  # Update every simulation step
    )

    # Lights
    sky_light = sim_utils.DomeLightCfg(
        intensity=750.0,
        color=(0.9, 0.9, 0.9),
    )


@configclass
class DigitRoughSceneCfg(DigitSceneCfg):
    """Scene configuration with rough terrain for robust training."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=sim_utils.TerrainGeneratorCfg(
            seed=42,
            size=(8.0, 8.0),
            border_width=20.0,
            num_rows=10,
            num_cols=20,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
            curriculum=True,
            difficulty_range=(0.0, 1.0),
            sub_terrains={
                "flat": sim_utils.FlatTerrainCfg(proportion=0.2),
                "random_rough": sim_utils.HfRandomUniformTerrainCfg(
                    proportion=0.3,
                    noise_range=(0.01, 0.06),
                    noise_step=0.01,
                ),
                "pyramid_stairs": sim_utils.MeshPyramidStairsTerrainCfg(
                    proportion=0.25,
                    step_height_range=(0.05, 0.15),
                    step_width=0.3,
                    platform_width=3.0,
                ),
                "pyramid_stairs_inv": sim_utils.MeshInvertedPyramidStairsTerrainCfg(
                    proportion=0.25,
                    step_height_range=(0.05, 0.15),
                    step_width=0.3,
                    platform_width=3.0,
                ),
            },
        ),
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )


# =============================================================================
# COMMANDS CONFIGURATION
# =============================================================================

@configclass
class CommandsCfg:
    """Configuration for velocity commands."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,  # 2% of envs commanded to stand still
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),      # m/s forward/backward
            lin_vel_y=(-0.5, 0.5),      # m/s lateral
            ang_vel_z=(-1.0, 1.0),      # rad/s yaw
            heading=(-math.pi, math.pi),
        ),
    )


# =============================================================================
# ACTIONS CONFIGURATION
# =============================================================================

@configclass
class ActionsCfg:
    """Action space configuration.

    Uses joint position control where actions are position offsets
    from the default standing pose.
    """

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],  # All actuated joints
        scale=0.25,          # Scale factor for position offsets
        use_default_offset=True,  # Actions are relative to default pose
    )


# =============================================================================
# OBSERVATIONS CONFIGURATION
# =============================================================================

@configclass
class ObservationsCfg:
    """Observation space configuration.

    Observations are grouped for the policy network. Additional groups
    can be added for critics or auxiliary tasks.
    """

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        """Observations provided to the policy network."""

        # Velocity commands (what the robot should do)
        velocity_commands = ObservationTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )

        # Base state (proprioception)
        base_lin_vel = ObservationTermCfg(
            func=mdp.base_lin_vel,
            noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.05),
        )
        base_ang_vel = ObservationTermCfg(
            func=mdp.base_ang_vel,
            noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.1),
        )
        projected_gravity = ObservationTermCfg(
            func=mdp.projected_gravity,
            noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.025),
        )

        # Joint state
        joint_pos = ObservationTermCfg(
            func=mdp.joint_pos_rel,
            noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.01),
        )
        joint_vel = ObservationTermCfg(
            func=mdp.joint_vel_rel,
            noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.5),
        )

        # Previous actions (temporal context)
        actions = ObservationTermCfg(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # Instantiate observation groups
    policy: PolicyCfg = PolicyCfg()


# =============================================================================
# REWARDS CONFIGURATION
# =============================================================================

@configclass
class RewardsCfg:
    """Reward function configuration.

    Organized into categories:
    - Tracking: Primary objectives
    - Stability: Keep robot upright
    - Gait: Encourage proper walking
    - Regularization: Smooth, efficient motion
    """

    # === Tracking Rewards (Primary Objectives) ===
    track_lin_vel_xy_exp = RewardTermCfg(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z_exp = RewardTermCfg(
        func=mdp.track_ang_vel_z_exp,
        weight=0.75,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )

    # === Stability Penalties ===
    lin_vel_z_l2 = RewardTermCfg(
        func=mdp.lin_vel_z_l2,
        weight=-2.0,
    )
    ang_vel_xy_l2 = RewardTermCfg(
        func=mdp.ang_vel_xy_l2,
        weight=-0.05,
    )
    flat_orientation_l2 = RewardTermCfg(
        func=mdp.flat_orientation_l2,
        weight=-1.0,
    )

    # === Gait Quality ===
    feet_air_time = RewardTermCfg(
        func=mdp.feet_air_time,
        weight=0.125,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*toe.*"),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )

    # === Regularization Penalties ===
    action_rate_l2 = RewardTermCfg(
        func=mdp.action_rate_l2,
        weight=-0.01,
    )
    joint_acc_l2 = RewardTermCfg(
        func=mdp.joint_acc_l2,
        weight=-2.5e-7,
    )
    joint_torques_l2 = RewardTermCfg(
        func=mdp.applied_torque_limits,
        weight=-1e-5,
    )

    # === Safety Penalties ===
    undesired_contacts = RewardTermCfg(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[".*shin.*", ".*thigh.*", ".*torso.*", ".*pelvis.*"],
            ),
            "threshold": 1.0,
        },
    )
    joint_pos_limits = RewardTermCfg(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


# =============================================================================
# TERMINATIONS CONFIGURATION
# =============================================================================

@configclass
class TerminationsCfg:
    """Episode termination conditions."""

    # Time limit
    time_out = TerminationTermCfg(
        func=mdp.time_out,
        time_out=True,
    )

    # Fall detection - contact on torso/pelvis
    base_contact = TerminationTermCfg(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[".*torso.*", ".*pelvis.*"],
            ),
            "threshold": 1.0,
        },
    )

    # Excessive tilt
    bad_orientation = TerminationTermCfg(
        func=mdp.bad_orientation,
        params={"limit_angle": 0.5},  # ~30 degrees
    )


# =============================================================================
# EVENTS CONFIGURATION (Domain Randomization)
# =============================================================================

@configclass
class EventsCfg:
    """Domain randomization events for sim-to-real transfer."""

    # === Reset Events ===
    reset_base = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "yaw": (-math.pi, math.pi),
            },
            "velocity_range": {
                "x": (-0.3, 0.3),
                "y": (-0.3, 0.3),
                "z": (0.0, 0.0),
                "roll": (-0.1, 0.1),
                "pitch": (-0.1, 0.1),
                "yaw": (-0.3, 0.3),
            },
        },
    )

    reset_joints = EventTermCfg(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.1, 0.1),
            "velocity_range": (-0.5, 0.5),
        },
    )

    # === Interval Events (During Episode) ===
    push_robot = EventTermCfg(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
            },
        },
    )


@configclass
class EventsRoughCfg(EventsCfg):
    """Extended domain randomization for rough terrain training."""

    # === Startup Events (Once per environment) ===
    randomize_friction = EventTermCfg(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.6, 1.4),
            "dynamic_friction_range": (0.6, 1.4),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    randomize_mass = EventTermCfg(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.9, 1.1),
            "operation": "scale",
        },
    )

    # === Reset Events ===
    randomize_actuator_gains = EventTermCfg(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    # Stronger pushes for robust training
    push_robot = EventTermCfg(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(8.0, 12.0),
        params={
            "velocity_range": {
                "x": (-0.8, 0.8),
                "y": (-0.8, 0.8),
            },
        },
    )


# =============================================================================
# CURRICULUM CONFIGURATION
# =============================================================================

@configclass
class CurriculumCfg:
    """Curriculum learning configuration."""

    terrain_levels = CurriculumTermCfg(
        func=mdp.terrain_levels_vel,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


# =============================================================================
# COMPLETE ENVIRONMENT CONFIGURATIONS
# =============================================================================

@configclass
class DigitFlatEnvCfg(ManagerBasedRLEnvCfg):
    """Complete environment configuration for Digit on flat terrain.

    Recommended for initial training to establish basic walking.
    """

    # Scene
    scene: DigitSceneCfg = DigitSceneCfg(num_envs=4096, env_spacing=2.5)

    # MDP components
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()

    # Simulation settings
    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=0.005,  # 200 Hz physics
        render_interval=4,  # 50 Hz rendering
        gravity=(0.0, 0.0, -9.81),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )

    # Episode settings
    episode_length_s = 20.0
    decimation = 4  # Policy at 50 Hz (200/4)

    def __post_init__(self):
        """Post-initialization configuration."""
        super().__post_init__()

        # Ensure consistent dt and decimation
        self.sim.dt = 0.005
        self.decimation = 4

        # Disable curriculum for flat terrain
        self.curriculum = None


@configclass
class DigitRoughEnvCfg(DigitFlatEnvCfg):
    """Environment configuration for Digit on rough terrain.

    Extends flat environment with:
    - Varied terrain types
    - Additional domain randomization
    - Terrain curriculum
    """

    # Override scene with rough terrain
    scene: DigitRoughSceneCfg = DigitRoughSceneCfg(num_envs=4096, env_spacing=2.5)

    # Extended domain randomization
    events: EventsRoughCfg = EventsRoughCfg()

    # Enable terrain curriculum
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post-initialization configuration."""
        super().__post_init__()

        # Longer episodes for terrain navigation
        self.episode_length_s = 30.0
