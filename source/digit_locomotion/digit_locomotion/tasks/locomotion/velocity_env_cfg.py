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
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
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
from isaaclab.terrains import (
    TerrainImporterCfg,
    TerrainGeneratorCfg,
    HfRandomUniformTerrainCfg,
    MeshPlaneTerrainCfg,
    MeshPyramidStairsTerrainCfg,
    MeshInvertedPyramidStairsTerrainCfg,
)
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg, AdditiveGaussianNoiseCfg

# Import MDP components (using locomotion-specific mdp with extra reward functions)
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

# Import custom MDP components (arm swing coordination, etc.)
from . import mdp as custom_mdp

# Import Digit robot configuration from isaaclab_assets
from isaaclab_assets.robots.agility import DIGIT_V4_CFG as DIGIT_CFG


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
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            color=(0.9, 0.9, 0.9),
        ),
    )


@configclass
class DigitRoughSceneCfg(DigitSceneCfg):
    """Scene configuration with rough terrain for robust training."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
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
                "flat": MeshPlaneTerrainCfg(proportion=0.2),
                "random_rough": HfRandomUniformTerrainCfg(
                    proportion=0.3,
                    noise_range=(0.01, 0.06),
                    noise_step=0.01,
                ),
                "pyramid_stairs": MeshPyramidStairsTerrainCfg(
                    proportion=0.25,
                    step_height_range=(0.05, 0.15),
                    step_width=0.3,
                    platform_width=3.0,
                ),
                "pyramid_stairs_inv": MeshInvertedPyramidStairsTerrainCfg(
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
# CURRICULUM PHASE COMMANDS
# =============================================================================
# Each phase has its own velocity range. Use different --task to switch phases.
# Resume from checkpoint when progressing to the next phase.


@configclass
class CommandsWalkingCfg:
    """Phase 1: Walking (0-2 m/s) - Learn balance and basic locomotion."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(8.0, 12.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 2.0),         # Walking speed
            lin_vel_y=(-0.2, 0.2),
            ang_vel_z=(-0.3, 0.3),
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class CommandsJoggingIntroCfg:
    """Phase 1.75: Jogging Intro (0-3 m/s) - Bridge to faster jogging."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(8.0, 12.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 3.0),         # Bridge speed (0-3 m/s)
            lin_vel_y=(-0.2, 0.2),
            ang_vel_z=(-0.3, 0.3),
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class CommandsJoggingCfg:
    """Phase 2: Jogging (0-5 m/s) - Transition to running gait."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(8.0, 12.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 5.0),         # Jogging speed
            lin_vel_y=(-0.2, 0.2),
            ang_vel_z=(-0.3, 0.3),
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class CommandsRunningCfg:
    """Phase 3: Running (0-8 m/s) - Fast running gait."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(8.0, 12.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 8.0),         # Running speed
            lin_vel_y=(-0.2, 0.2),
            ang_vel_z=(-0.3, 0.3),
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class CommandsFastRunningCfg:
    """Phase 4: Fast Running (0-10 m/s) - Sprint warmup."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(8.0, 12.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 10.0),        # Fast running speed
            lin_vel_y=(-0.2, 0.2),
            ang_vel_z=(-0.3, 0.3),
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class CommandsSprintCfg:
    """Phase 5: Sprint (0-13.5 m/s) - Full speed 30 mph."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(8.0, 12.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 13.5),        # Sprint speed (30 mph)
            lin_vel_y=(-0.2, 0.2),
            ang_vel_z=(-0.3, 0.3),
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


@configclass
class ActionsMinimalCfg:
    """Minimal action space - legs only (8 DOF).

    Controls only the primary leg joints for faster training:
    - hip_roll, hip_yaw, hip_pitch, knee (4 per leg = 8 total)

    Arms and passive leg joints (shin, tarsus, toe) are fixed.
    This significantly reduces:
    - Action space dimensionality (50 -> 8)
    - Observation space (fewer joint states)
    - Training time and complexity
    """

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            ".*hip_roll",
            ".*hip_yaw",
            ".*hip_pitch",
            ".*knee",
        ],  # Only main leg joints (8 DOF total)
        scale=0.25,
        use_default_offset=True,
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
            # Use toe_roll as the foot contact point for Digit V4
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_leg_toe_roll"),
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
        func=mdp.joint_torques_l2,
        weight=-1e-6,  # Same as Isaac Lab's Digit config
    )

    # === Safety Penalties ===
    undesired_contacts = RewardTermCfg(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                # Updated for Digit V4 body names: penalize contact on rods and tarsus
                body_names=[".*_rod", ".*tarsus"],
            ),
            "threshold": 1.0,
        },
    )
    joint_pos_limits = RewardTermCfg(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class RewardsRunningCfg:
    """Reward function configuration for running gait with natural arm swing.

    Includes arm swing coordination reward to encourage human-like movement:
    - Arms swing in opposition to legs (left leg forward -> right arm forward)
    - Arms stay close to body (no lateral flailing)
    - Provides balance and is more energy efficient

    Key features:
    - Arm swing coordination reward for natural movement
    - Arm posture rewards to keep arms close to torso
    - Slight penalty for deviation from default joint positions
    - Standard locomotion rewards for velocity tracking and stability
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

    # === Arm Swing Coordination (Natural Movement) ===
    arm_swing = RewardTermCfg(
        func=custom_mdp.arm_swing_coordination,
        weight=0.3,  # Moderate weight to encourage but not dominate
        params={"command_name": "base_velocity"},
    )

    # === Arm Posture Rewards (Keep Arms Close to Body) ===
    arm_close_to_body = RewardTermCfg(
        func=custom_mdp.arm_close_to_body,
        weight=0.2,  # Reward keeping arms close to torso
        params={"command_name": "base_velocity"},
    )
    arm_lateral_penalty = RewardTermCfg(
        func=custom_mdp.arm_lateral_penalty,
        weight=-0.4,  # Penalize arms extending sideways
        params={
            "command_name": "base_velocity",
            "max_lateral": 0.3,  # ~17 degrees allowed
        },
    )

    # === Default Pose Penalty (Prevent Extreme Arm Positions) ===
    joint_default = RewardTermCfg(
        func=custom_mdp.joint_default_position,
        weight=-0.08,  # Increased for stronger posture enforcement
        params={"asset_cfg": SceneEntityCfg("robot")},
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
        weight=-0.5,
    )

    # === Gait Quality ===
    feet_air_time = RewardTermCfg(
        func=mdp.feet_air_time,
        weight=0.125,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_leg_toe_roll"),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )

    # === Regularization Penalties ===
    action_rate_l2 = RewardTermCfg(
        func=mdp.action_rate_l2,
        weight=-0.01,  # Same as walking
    )
    joint_acc_l2 = RewardTermCfg(
        func=mdp.joint_acc_l2,
        weight=-2.5e-7,  # Same as walking
    )
    joint_torques_l2 = RewardTermCfg(
        func=mdp.joint_torques_l2,
        weight=-1e-6,  # Same as Isaac Lab's Digit config
    )

    # === Safety Penalties ===
    undesired_contacts = RewardTermCfg(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[".*_rod", ".*tarsus"],
            ),
            "threshold": 1.0,
        },
    )
    joint_pos_limits = RewardTermCfg(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class RewardsJoggingCfg:
    """Reward function configuration for jogging with improved posture.

    Extends running rewards with:
    - Forward lean proportional to speed (like human jogging)
    - Bent elbows for efficient arm swing
    - Arms close to body (no lateral flailing)
    - Relaxed flat orientation penalty (allows natural body movement)
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

    # === Arm Swing Coordination (Natural Movement) ===
    arm_swing = RewardTermCfg(
        func=custom_mdp.arm_swing_coordination,
        weight=0.3,
        params={"command_name": "base_velocity"},
    )

    # === Jogging Posture Rewards ===
    forward_lean = RewardTermCfg(
        func=custom_mdp.forward_lean_reward,
        weight=0.2,  # Encourage forward lean when moving fast
        params={
            "command_name": "base_velocity",
            "target_lean_per_speed": 0.04,  # ~2.3 degrees per m/s
            "max_lean": 0.15,  # ~8.5 degrees max
        },
    )
    elbow_bend = RewardTermCfg(
        func=custom_mdp.elbow_bend_while_moving,
        weight=0.15,  # Encourage bent elbows
        params={
            "command_name": "base_velocity",
            "target_bend": 0.8,  # ~45 degrees
        },
    )

    # === Arm Posture Rewards (Keep Arms Close to Body) ===
    arm_close_to_body = RewardTermCfg(
        func=custom_mdp.arm_close_to_body,
        weight=0.25,  # Reward keeping arms close to torso
        params={"command_name": "base_velocity"},
    )
    arm_lateral_penalty = RewardTermCfg(
        func=custom_mdp.arm_lateral_penalty,
        weight=-0.5,  # Penalize arms extending sideways
        params={
            "command_name": "base_velocity",
            "max_lateral": 0.3,  # ~17 degrees allowed
        },
    )

    # === Default Pose Penalty (Increased for better arm control) ===
    joint_default = RewardTermCfg(
        func=custom_mdp.joint_default_position,
        weight=-0.08,  # Increased from -0.02 for stronger posture enforcement
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # === Stability Penalties (Relaxed for jogging) ===
    lin_vel_z_l2 = RewardTermCfg(
        func=mdp.lin_vel_z_l2,
        weight=-1.5,  # Reduced from -2.0 to allow more vertical movement
    )
    ang_vel_xy_l2 = RewardTermCfg(
        func=mdp.ang_vel_xy_l2,
        weight=-0.03,  # Reduced from -0.05 to allow more body rotation
    )
    flat_orientation_l2 = RewardTermCfg(
        func=mdp.flat_orientation_l2,
        weight=-0.2,  # Reduced from -0.5 to allow forward lean
    )

    # === Gait Quality ===
    feet_air_time = RewardTermCfg(
        func=mdp.feet_air_time,
        weight=0.15,  # Slightly higher for jogging (more air time expected)
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_leg_toe_roll"),
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
        func=mdp.joint_torques_l2,
        weight=-1e-6,
    )

    # === Safety Penalties ===
    undesired_contacts = RewardTermCfg(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[".*_rod", ".*tarsus"],
            ),
            "threshold": 1.0,
        },
    )
    joint_pos_limits = RewardTermCfg(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class RewardsJoggingScheduledCfg:
    """Reward configuration for jogging with SCHEDULED arm posture rewards.

    Use this with --reward_schedule jogging to gradually introduce arm posture
    rewards after locomotion has stabilized.

    Key differences from RewardsJoggingCfg:
    - arm_close_to_body: starts at 0 (scheduler ramps to 0.25)
    - arm_lateral_penalty: starts at 0 (scheduler ramps to -0.5)
    - elbow_bend: starts at 0 (scheduler ramps to 0.15)
    - forward_lean: starts at 0 (scheduler ramps to 0.2)
    - joint_default: starts at -0.02 (scheduler ramps to -0.08)

    This allows the policy to first learn stable locomotion, then gradually
    adapt to arm posture constraints without catastrophic forgetting.
    """

    # === Tracking Rewards (Primary Objectives) - Full weight from start ===
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

    # === Arm Swing Coordination - Keep this active ===
    arm_swing = RewardTermCfg(
        func=custom_mdp.arm_swing_coordination,
        weight=0.3,
        params={"command_name": "base_velocity"},
    )

    # === SCHEDULED: Jogging Posture Rewards (start at 0) ===
    forward_lean = RewardTermCfg(
        func=custom_mdp.forward_lean_reward,
        weight=0.0,  # SCHEDULED: 0 -> 0.2
        params={
            "command_name": "base_velocity",
            "target_lean_per_speed": 0.04,
            "max_lean": 0.15,
        },
    )
    elbow_bend = RewardTermCfg(
        func=custom_mdp.elbow_bend_while_moving,
        weight=0.0,  # SCHEDULED: 0 -> 0.15
        params={
            "command_name": "base_velocity",
            "target_bend": 0.8,
        },
    )

    # === SCHEDULED: Arm Posture Rewards (start at 0) ===
    arm_close_to_body = RewardTermCfg(
        func=custom_mdp.arm_close_to_body,
        weight=0.0,  # SCHEDULED: 0 -> 0.25
        params={"command_name": "base_velocity"},
    )
    arm_lateral_penalty = RewardTermCfg(
        func=custom_mdp.arm_lateral_penalty,
        weight=0.0,  # SCHEDULED: 0 -> -0.5
        params={
            "command_name": "base_velocity",
            "max_lateral": 0.3,
        },
    )

    # === SCHEDULED: Default Pose Penalty (start low) ===
    joint_default = RewardTermCfg(
        func=custom_mdp.joint_default_position,
        weight=-0.02,  # SCHEDULED: -0.02 -> -0.08
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # === Stability Penalties (Relaxed for jogging) - Active from start ===
    lin_vel_z_l2 = RewardTermCfg(
        func=mdp.lin_vel_z_l2,
        weight=-1.5,
    )
    ang_vel_xy_l2 = RewardTermCfg(
        func=mdp.ang_vel_xy_l2,
        weight=-0.03,
    )
    flat_orientation_l2 = RewardTermCfg(
        func=mdp.flat_orientation_l2,
        weight=-0.2,
    )

    # === Gait Quality - Active from start ===
    feet_air_time = RewardTermCfg(
        func=mdp.feet_air_time,
        weight=0.15,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_leg_toe_roll"),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )

    # === Regularization Penalties - Active from start ===
    action_rate_l2 = RewardTermCfg(
        func=mdp.action_rate_l2,
        weight=-0.01,
    )
    joint_acc_l2 = RewardTermCfg(
        func=mdp.joint_acc_l2,
        weight=-2.5e-7,
    )
    joint_torques_l2 = RewardTermCfg(
        func=mdp.joint_torques_l2,
        weight=-1e-6,
    )

    # === Safety Penalties - Active from start ===
    undesired_contacts = RewardTermCfg(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[".*_rod", ".*tarsus"],
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

    # Fall detection - contact on torso
    base_contact = TerminationTermCfg(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                # Updated for Digit V4: only torso_base for fall detection
                body_names=["torso_base"],
            ),
            "threshold": 1.0,
        },
    )

    # Excessive tilt
    bad_orientation = TerminationTermCfg(
        func=mdp.bad_orientation,
        params={"limit_angle": 0.5},  # ~30 degrees
    )


@configclass
class TerminationsRunningCfg:
    """Episode termination conditions for running.

    PHASE 1: Same as walking to learn stable gait first.
    PHASE 2 (later): Increase limit_angle to allow more forward lean.
    """

    # Time limit
    time_out = TerminationTermCfg(
        func=mdp.time_out,
        time_out=True,
    )

    # Fall detection - contact on torso
    base_contact = TerminationTermCfg(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["torso_base"],
            ),
            "threshold": 1.0,
        },
    )

    # Excessive tilt - same as walking initially
    bad_orientation = TerminationTermCfg(
        func=mdp.bad_orientation,
        params={"limit_angle": 0.5},  # ~30 degrees (same as walking)
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

    GPU Optimization: Default num_envs=8192 for better GPU utilization.
    On RTX 4090 (24GB), you can use up to 16384 envs.
    """

    # Scene - 8192 envs for better GPU utilization (was 4096)
    scene: DigitSceneCfg = DigitSceneCfg(num_envs=8192, env_spacing=2.5)

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

    # Override scene with rough terrain - 8192 envs for better GPU utilization
    scene: DigitRoughSceneCfg = DigitRoughSceneCfg(num_envs=8192, env_spacing=2.5)

    # Extended domain randomization
    events: EventsRoughCfg = EventsRoughCfg()

    # Enable terrain curriculum
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post-initialization configuration."""
        super().__post_init__()

        # Longer episodes for terrain navigation
        self.episode_length_s = 30.0


@configclass
class DigitMinimalEnvCfg(DigitFlatEnvCfg):
    """Minimal environment for fast training experiments.

    Uses reduced action space (8 DOF legs only) for:
    - Faster training iterations
    - Quicker experimentation with rewards/hyperparameters
    - Proof-of-concept before full training

    DOF Comparison:
    - Full: 50 joints (all body)
    - Minimal: 8 joints (hip_roll, hip_yaw, hip_pitch, knee x2)

    Training speedup comes from:
    - Smaller action/observation spaces
    - Simpler policy to learn
    - Faster physics (fewer active joints)

    GPU Optimization:
    - 16384 envs (2x more than full config - less memory per env)
    - Combined with higher num_steps_per_env in PPO config
    """

    # Balanced environment count for fast iterations
    scene: DigitSceneCfg = DigitSceneCfg(num_envs=16384, env_spacing=2.5)

    # Use minimal actions (legs only)
    actions: ActionsMinimalCfg = ActionsMinimalCfg()

    def __post_init__(self):
        """Post-initialization configuration."""
        super().__post_init__()

        # Shorter episodes for faster iteration during experiments
        self.episode_length_s = 10.0


# =============================================================================
# CURRICULUM PHASE ENVIRONMENT CONFIGURATIONS
# =============================================================================
# Use different --task to switch between phases:
#   Phase 1: Digit-Walking-v0      (0-2 m/s)
#   Phase 2: Digit-Jogging-v0      (0-5 m/s)
#   Phase 3: Digit-Running-v0      (0-8 m/s)
#   Phase 4: Digit-FastRunning-v0  (0-10 m/s)
#   Phase 5: Digit-Sprint-v0       (0-13.5 m/s / 30 mph)
#
# Resume from checkpoint when progressing: --checkpoint <path_to_model.pt>


@configclass
class DigitWalkingEnvCfg(DigitFlatEnvCfg):
    """Phase 1: Walking (0-2 m/s) - Learn balance and basic locomotion.

    Start here for initial training. Features:
    - Full body control (50 DOF including arms)
    - Arm swing coordination reward
    - Conservative velocity range for learning balance
    """

    scene: DigitSceneCfg = DigitSceneCfg(num_envs=16384, env_spacing=2.5)
    commands: CommandsWalkingCfg = CommandsWalkingCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsRunningCfg = RewardsRunningCfg()
    terminations: TerminationsRunningCfg = TerminationsRunningCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 15.0


@configclass
class DigitJoggingWarmupEnvCfg(DigitFlatEnvCfg):
    """Phase 1.5: Jogging Warmup (0-2 m/s) - Learn jogging posture at walking speed.

    Train from scratch to learn jogging rewards (forward lean, elbow bend)
    at conservative walking speeds. Features:
    - Walking velocity range (0-2 m/s) for stability
    - Forward lean reward (lean proportional to speed)
    - Elbow bend reward (bent arms like human jogging)
    - Relaxed orientation penalty (allows natural body movement)

    Use this checkpoint to transfer to full jogging (0-5 m/s).
    """

    scene: DigitSceneCfg = DigitSceneCfg(num_envs=16384, env_spacing=2.5)
    commands: CommandsWalkingCfg = CommandsWalkingCfg()  # Walking speed (0-2 m/s)
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsJoggingCfg = RewardsJoggingCfg()  # Jogging posture rewards
    terminations: TerminationsRunningCfg = TerminationsRunningCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 15.0


@configclass
class DigitJoggingIntroEnvCfg(DigitFlatEnvCfg):
    """Phase 1.75: Jogging Intro (0-3 m/s) - Bridge to faster jogging.

    Resume from JoggingWarmup checkpoint. Features:
    - Intermediate velocity (0-3 m/s) to bridge 0-2 and 0-5
    - Forward lean + elbow bend rewards
    - Relaxed orientation penalty

    Use this checkpoint to transfer to full jogging (0-5 m/s).
    """

    scene: DigitSceneCfg = DigitSceneCfg(num_envs=16384, env_spacing=2.5)
    commands: CommandsJoggingIntroCfg = CommandsJoggingIntroCfg()  # 0-3 m/s
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsJoggingCfg = RewardsJoggingCfg()
    terminations: TerminationsRunningCfg = TerminationsRunningCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 15.0


@configclass
class DigitJoggingIntroScheduledEnvCfg(DigitFlatEnvCfg):
    """Phase 1.75: Jogging Intro with SCHEDULED rewards.

    Use this with --reward_schedule jogging to gradually introduce
    arm posture rewards. This prevents catastrophic forgetting when
    resuming from a walking/jogging checkpoint.

    Training command:
        .\\isaaclab.bat -p scripts\\train.py --task Digit-JoggingIntroScheduled-v0 \\
            --checkpoint <checkpoint.pt> --reward_schedule jogging --headless

    The scheduler will:
    1. First 200 iterations: locomotion only (arm rewards = 0)
    2. Iterations 200-500: gradually introduce arm posture rewards
    3. After iteration 500: full arm posture rewards active
    """

    scene: DigitSceneCfg = DigitSceneCfg(num_envs=16384, env_spacing=2.5)
    commands: CommandsJoggingIntroCfg = CommandsJoggingIntroCfg()  # 0-3 m/s
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsJoggingScheduledCfg = RewardsJoggingScheduledCfg()  # Scheduled!
    terminations: TerminationsRunningCfg = TerminationsRunningCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 15.0


@configclass
class DigitJoggingEnvCfg(DigitFlatEnvCfg):
    """Phase 2: Jogging (0-5 m/s) - Transition to running gait.

    Resume from JoggingIntro checkpoint. Features:
    - Higher velocity commands (0-5 m/s)
    - Forward lean reward (lean proportional to speed)
    - Elbow bend reward (bent arms like human jogging)
    - Relaxed orientation penalty (allows natural body movement)
    """

    scene: DigitSceneCfg = DigitSceneCfg(num_envs=16384, env_spacing=2.5)
    commands: CommandsJoggingCfg = CommandsJoggingCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsJoggingCfg = RewardsJoggingCfg()
    terminations: TerminationsRunningCfg = TerminationsRunningCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 15.0


@configclass
class DigitRunningEnvCfg(DigitFlatEnvCfg):
    """Phase 3: Running (0-8 m/s) - Fast running gait.

    Resume from Phase 2 checkpoint. Features:
    - Fast running velocities (0-8 m/s)
    - Forward lean + elbow bend for natural posture
    - Full body control with arm swing
    """

    scene: DigitSceneCfg = DigitSceneCfg(num_envs=16384, env_spacing=2.5)
    commands: CommandsRunningCfg = CommandsRunningCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsJoggingCfg = RewardsJoggingCfg()  # Same posture rewards
    terminations: TerminationsRunningCfg = TerminationsRunningCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 15.0


@configclass
class DigitFastRunningEnvCfg(DigitFlatEnvCfg):
    """Phase 4: Fast Running (0-10 m/s) - Sprint warmup.

    Resume from Phase 3 checkpoint. Features:
    - Near-sprint velocities (0-10 m/s)
    - Forward lean + elbow bend for natural posture
    - Prepares policy for full sprint
    """

    scene: DigitSceneCfg = DigitSceneCfg(num_envs=16384, env_spacing=2.5)
    commands: CommandsFastRunningCfg = CommandsFastRunningCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsJoggingCfg = RewardsJoggingCfg()  # Same posture rewards
    terminations: TerminationsRunningCfg = TerminationsRunningCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 15.0


@configclass
class DigitSprintEnvCfg(DigitFlatEnvCfg):
    """Phase 5: Sprint (0-13.5 m/s / 30 mph) - Full speed sprint.

    Resume from Phase 4 checkpoint. Features:
    - Maximum velocity (30 mph / 13.5 m/s)
    - Forward lean + elbow bend for natural posture
    - Full body control with optimized arm swing
    """

    scene: DigitSceneCfg = DigitSceneCfg(num_envs=16384, env_spacing=2.5)
    commands: CommandsSprintCfg = CommandsSprintCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsJoggingCfg = RewardsJoggingCfg()  # Same posture rewards
    terminations: TerminationsRunningCfg = TerminationsRunningCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 15.0
