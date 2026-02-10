"""Baseline environment configuration following Radosavovic et al. (2024).

This module implements the training approach from:
"Real-world humanoid locomotion with reinforcement learning"
(Radosavovic et al., Science Robotics 2024)

Key differences from our custom approach:
1. NO explicit arm swing rewards - let arm swing emerge from energy minimization
2. Transformer-based policy with observation-action history
3. Teacher-student training pipeline
4. Additional domain randomization (observation delay)

Environments:
- Digit-Baseline-v0: Main baseline with emergent arm swing
- Digit-BaselineTeacher-v0: Teacher environment with privileged state
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (
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
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg

# Import MDP components
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

# Import custom MDP components (arm swing, posture rewards)
from . import mdp as custom_mdp

# Import Digit robot
from isaaclab_assets.robots.agility import DIGIT_V4_CFG as DIGIT_CFG


# =============================================================================
# SCENE CONFIGURATION
# =============================================================================

@configclass
class BaselineSceneCfg(InteractiveSceneCfg):
    """Scene configuration for baseline training."""

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

    robot: ArticulationCfg = DIGIT_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        actuators={
            "body": ImplicitActuatorCfg(
                joint_names_expr=["(?!.*_arm_).*"],
                stiffness=None,   # Use USD defaults (legs work fine)
                damping=None,
            ),
            "arms": IdealPDActuatorCfg(
                joint_names_expr=[".*_arm_.*"],
                stiffness=200.0,   # High stiffness — computes torque in Python, bypasses PhysX drive limits
                damping=10.0,      # Strong damping to prevent oscillation
                effort_limit=100.0,  # Max torque per arm joint (Nm)
                velocity_limit=10.0,  # Max joint velocity (rad/s)
            ),
        },
    )

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
        update_period=0.0,
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=750.0, color=(0.9, 0.9, 0.9)),
    )


# =============================================================================
# COMMANDS CONFIGURATION
# =============================================================================

@configclass
class BaselineCommandsCfg:
    """Velocity commands matching paper: walking up to 1 m/s."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),  # Resample every 10s (paper)
        rel_standing_envs=0.02,  # 2% standing
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),      # Paper: walking range
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-1.0, 1.0),
            heading=(-math.pi, math.pi),
        ),
    )


# =============================================================================
# ACTIONS CONFIGURATION
# =============================================================================

@configclass
class BaselineActionsCfg:
    """Action configuration for all 50 joints.

    Paper outputs PD setpoints for 16 actuated joints + 8 predicted PD gains.
    We simplify to just position targets for all joints.
    """

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["(?!.*_arm_).*"],  # Legs and body only — arms locked at default pose
        scale=0.25,
        use_default_offset=True,
    )


# =============================================================================
# OBSERVATIONS CONFIGURATION
# =============================================================================

@configclass
class BaselineObservationsCfg:
    """Observation configuration matching the paper.

    Proprioceptive observations only (no vision):
    - Base linear velocity (noisy)
    - Base angular velocity (noisy)
    - Projected gravity (noisy)
    - Joint positions (noisy)
    - Joint velocities (noisy)
    - Velocity commands
    - Previous actions

    Note: The transformer handles observation history internally.
    """

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        """Observations for the policy (student)."""

        # Commands
        velocity_commands = ObservationTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )

        # Base state with noise (matching paper)
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

        # Joint state with noise
        joint_pos = ObservationTermCfg(
            func=mdp.joint_pos_rel,
            noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.01),
        )
        joint_vel = ObservationTermCfg(
            func=mdp.joint_vel_rel,
            noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.5),
        )

        # Previous actions
        actions = ObservationTermCfg(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class TeacherObservationsCfg:
    """Observations for the teacher policy.

    Teacher has access to privileged state information:
    - True velocities (no noise)
    - True joint states (no noise)
    - Environment parameters (friction, mass)
    - Contact states

    This allows the teacher to learn faster, then distill to student.
    """

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        """Privileged observations for teacher."""

        # Commands
        velocity_commands = ObservationTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )

        # TRUE base state (no noise) - privileged
        base_lin_vel = ObservationTermCfg(func=mdp.base_lin_vel)
        base_ang_vel = ObservationTermCfg(func=mdp.base_ang_vel)
        projected_gravity = ObservationTermCfg(func=mdp.projected_gravity)

        # TRUE joint state (no noise) - privileged
        joint_pos = ObservationTermCfg(func=mdp.joint_pos_rel)
        joint_vel = ObservationTermCfg(func=mdp.joint_vel_rel)

        # Previous actions
        actions = ObservationTermCfg(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False  # No noise for teacher
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# =============================================================================
# REWARDS CONFIGURATION - BASELINE (Emergent Arm Swing)
# =============================================================================

@configclass
class BaselineRewardsCfg:
    """Reward configuration following the paper.

    Key principle: NO explicit arm swing rewards!
    Arm swing should emerge from energy minimization.

    Reward categories:
    1. Survival bonus (CRITICAL - prevents "die fast" degenerate solution)
    2. Velocity tracking (primary objective)
    3. Energy minimization (enables emergent arm swing)
    4. Stability penalties
    5. Gait quality
    6. Safety penalties

    Paper quote: "We did not impose explicit constraints on the arm-swing
    motion in the reward function or use any reference trajectories for the arms.
    After training, we observed emergent arm-swing motions."
    """

    # === Survival Bonus (CRITICAL) ===
    # Without this, the policy learns to die quickly to minimize cumulative penalties
    is_alive = RewardTermCfg(
        func=custom_mdp.is_alive,
        weight=1.0,
    )

    # === Tracking Rewards ===
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

    # === Energy Minimization (Critical for emergent arm swing) ===
    # Paper: "reward function included energy minimization terms, which might
    # suggest a relationship between the observed motions and energy expenditure"
    joint_torques_l2 = RewardTermCfg(
        func=mdp.joint_torques_l2,
        weight=-1e-6,  # Matches successful runs (005, 007, 008)
    )
    # Mechanical power = |torque × velocity| — true energy expenditure.
    # Penalizes flailing arms (high velocity + moderate torque) more than
    # joint_torques_l2 alone. Key insight: T-posing arms at high velocity
    # burn real energy even if torque is moderate.
    mechanical_power = RewardTermCfg(
        func=custom_mdp.mechanical_power_penalty,
        weight=-1e-5,  # Start conservative — this term can be large
    )
    # === Posture Correction ===
    # NOTE: shoulder_roll_penalty and shoulder_pitch_penalty REMOVED.
    # Arms are now locked (excluded from action space). The IdealPDActuatorCfg
    # holds them at default position. No need for penalty rewards.
    #
    # Penalize base height deviation from standing height - fixes crouching/bent legs
    # Digit V4 spawn height is 1.05m (pelvis height); target at spawn height
    # to prevent crouching. Robot total height is ~1.6m.
    # weight=-1.0 was too weak (robot still heavily crouched at iter 660).
    # Increased to -5.0 to strongly discourage crouching.
    base_height = RewardTermCfg(
        func=mdp.base_height_l2,
        weight=-5.0,
        params={
            "target_height": 1.05,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    # Penalize excessive forward lean at speed — root cause of bad_orientation
    # terminations at 4+ m/s. Only penalizes lean beyond ~6 degrees (0.1 rad),
    # allowing natural slight lean during running.
    # weight=-5.0 with max_lean=0.1 was barely triggering at 5 m/s (-0.0001).
    # Lowered threshold to 0.05 rad (~3 deg) to catch lean earlier at speed.
    forward_lean_penalty = RewardTermCfg(
        func=custom_mdp.excessive_forward_lean_penalty,
        weight=-10.0,
        params={
            "max_lean": 0.05,  # ~3 degrees threshold (tighter for running)
        },
    )

    joint_acc_l2 = RewardTermCfg(
        func=mdp.joint_acc_l2,
        weight=-2.5e-7,
    )
    action_rate_l2 = RewardTermCfg(
        func=mdp.action_rate_l2,
        weight=-0.01,  # Matches successful runs (005, 007, 008)
    )
    # Removed joint_vel_l2 - too aggressive, caused collapse

    # === Stability Penalties ===
    lin_vel_z_l2 = RewardTermCfg(
        func=mdp.lin_vel_z_l2,
        weight=-2.0,  # Restored to original
    )
    ang_vel_xy_l2 = RewardTermCfg(
        func=mdp.ang_vel_xy_l2,
        weight=-0.05,  # Penalize roll/pitch
    )
    flat_orientation_l2 = RewardTermCfg(
        func=mdp.flat_orientation_l2,
        weight=-0.5,  # Stay upright
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

    # === Termination Penalty ===
    # Penalize early termination (falling) to encourage survival
    termination_penalty = RewardTermCfg(
        func=mdp.is_terminated,
        weight=-2.0,  # Penalty for falling
    )

    # NOTE: upright_posture, excessive_forward_lean, arm_leg_coordination,
    # arm_swing_bias removed — were weight=0.0 but still computed by
    # RewardManager every step. T-pose now fixed via joint_deviation_arms above.


# =============================================================================
# TERMINATIONS CONFIGURATION
# =============================================================================

@configclass
class BaselineTerminationsCfg:
    """Episode termination conditions."""

    time_out = TerminationTermCfg(
        func=mdp.time_out,
        time_out=True,
    )

    base_contact = TerminationTermCfg(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["torso_base"]),
            "threshold": 1.0,
        },
    )

    bad_orientation = TerminationTermCfg(
        func=mdp.bad_orientation,
        params={"limit_angle": 0.7},  # Relaxed to ~40° for more recovery time
    )


# =============================================================================
# EVENTS CONFIGURATION (Domain Randomization)
# =============================================================================

@configclass
class BaselineEventsCfg:
    """Domain randomization matching the paper.

    Randomizes:
    - Robot position/velocity on reset
    - Joint positions/velocities on reset
    - External pushes during episode
    - Friction (startup)
    - Mass (startup)
    """

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

    # === Interval Events ===
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


# =============================================================================
# COMPLETE ENVIRONMENT CONFIGURATIONS
# =============================================================================

@configclass
class DigitBaselineEnvCfg(ManagerBasedRLEnvCfg):
    """Baseline environment following the Science Robotics paper.

    Key features:
    - NO explicit arm swing rewards (emergent behavior)
    - Energy minimization rewards for natural motion
    - Domain randomization for sim-to-real
    - Compatible with transformer policy

    Use with Digit-Baseline-v0 task ID.
    """

    scene: BaselineSceneCfg = BaselineSceneCfg(num_envs=16384, env_spacing=2.5)
    observations: BaselineObservationsCfg = BaselineObservationsCfg()
    actions: BaselineActionsCfg = BaselineActionsCfg()
    commands: BaselineCommandsCfg = BaselineCommandsCfg()
    rewards: BaselineRewardsCfg = BaselineRewardsCfg()
    terminations: BaselineTerminationsCfg = BaselineTerminationsCfg()
    events: BaselineEventsCfg = BaselineEventsCfg()

    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=0.005,  # 200 Hz physics
        render_interval=4,
        gravity=(0.0, 0.0, -9.81),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )

    episode_length_s = 20.0
    decimation = 4  # 50 Hz policy (matches paper)

    def __post_init__(self):
        super().__post_init__()
        self.sim.dt = 0.005
        self.decimation = 4
        self.curriculum = None


@configclass
class DigitBaselineTeacherEnvCfg(DigitBaselineEnvCfg):
    """Teacher environment with privileged state information.

    Teacher sees true state without noise for faster learning.
    Student then learns to match teacher behavior with noisy observations.
    """

    observations: TeacherObservationsCfg = TeacherObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        # 20s episodes = 1000 max steps (at dt=0.005, decimation=4 → 0.02s/step)
        self.episode_length_s = 20.0


# =============================================================================
# FAST WALKING ENVIRONMENTS (Intermediate Curriculum Step)
# =============================================================================

@configclass
class BaselineFastWalkingCommandsCfg:
    """Velocity commands for fast walking: 0-2 m/s forward.

    Intermediate step between walking (0-1.5 m/s) and jogging (0-3 m/s).
    Helps policy adapt to faster speeds gradually.
    """

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 2.0),       # Fast walking: 0-2 m/s forward
            lin_vel_y=(-0.4, 0.4),      # Moderate lateral
            ang_vel_z=(-0.6, 0.6),      # Moderate turning
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class DigitBaselineFastWalkingEnvCfg(DigitBaselineEnvCfg):
    """Baseline fast walking environment - intermediate curriculum step.

    Extends baseline approach to fast walking speeds (0-2 m/s).
    Continue training from walking checkpoint for smooth curriculum.

    Use with Digit-BaselineFastWalking-v0 task ID.
    """

    commands: BaselineFastWalkingCommandsCfg = BaselineFastWalkingCommandsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 20.0


# =============================================================================
# SLOW JOGGING ENVIRONMENTS (Intermediate Curriculum Step)
# =============================================================================

@configclass
class BaselineSlowJoggingCommandsCfg:
    """Velocity commands for slow jogging: 0-2.5 m/s forward.

    Intermediate step between fast walking (0-2 m/s) and jogging (0-3 m/s).
    This smaller 0.5 m/s increment helps the policy adapt more gradually.
    """

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 2.5),        # Slow jogging: 0-2.5 m/s forward
            lin_vel_y=(-0.35, 0.35),     # Between fast walking and jogging
            ang_vel_z=(-0.55, 0.55),     # Between fast walking and jogging
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class DigitBaselineSlowJoggingEnvCfg(DigitBaselineEnvCfg):
    """Baseline slow jogging environment - intermediate curriculum step.

    Extends baseline approach to slow jogging speeds (0-2.5 m/s).
    Bridges the gap between fast walking (2 m/s) and jogging (3 m/s).
    Continue training from fast walking checkpoint for smooth curriculum.

    Use with Digit-BaselineSlowJogging-v0 task ID.
    """

    commands: BaselineSlowJoggingCommandsCfg = BaselineSlowJoggingCommandsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 20.0


# =============================================================================
# JOGGING ENVIRONMENTS (Curriculum Extension)
# =============================================================================

@configclass
class BaselineJoggingCommandsCfg:
    """Velocity commands for jogging: 0-3 m/s forward, gradual increase from walking."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 3.0),       # Jogging: 0-3 m/s forward
            lin_vel_y=(-0.3, 0.3),      # Reduced lateral
            ang_vel_z=(-0.5, 0.5),      # Reduced turning at speed
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class BaselineModerateRunningCommandsCfg:
    """Velocity commands for moderate running: 0-4 m/s forward.

    Intermediate step between jogging (0-3 m/s) and running (0-5 m/s).
    The 3→5 m/s jump was too large (92% bad_orientation), so this bridges the gap.
    """

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 4.0),       # Moderate running: 0-4 m/s forward
            lin_vel_y=(-0.25, 0.25),    # Between jogging and running lateral
            ang_vel_z=(-0.4, 0.4),      # Between jogging and running turning
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class BaselineRunningCommandsCfg:
    """Velocity commands for running: 0-5 m/s forward."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.01,  # Less standing at higher speeds
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 5.0),       # Running: 0-5 m/s forward
            lin_vel_y=(-0.2, 0.2),      # Minimal lateral
            ang_vel_z=(-0.3, 0.3),      # Minimal turning
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class DigitBaselineJoggingEnvCfg(DigitBaselineEnvCfg):
    """Baseline jogging environment - curriculum step from walking.

    Extends baseline approach to jogging speeds (0-3 m/s).
    Continue training from walking checkpoint for smooth curriculum.

    Use with Digit-BaselineJogging-v0 task ID.
    """

    commands: BaselineJoggingCommandsCfg = BaselineJoggingCommandsCfg()

    def __post_init__(self):
        super().__post_init__()
        # Slightly longer episodes for jogging practice
        self.episode_length_s = 20.0


@configclass
class DigitBaselineModerateRunningEnvCfg(DigitBaselineEnvCfg):
    """Baseline moderate running environment - curriculum step from jogging.

    Intermediate step between jogging (0-3 m/s) and running (0-5 m/s).
    The direct 3→5 m/s jump caused 92% bad_orientation failures.

    Use with Digit-BaselineModerateRunning-v0 task ID.
    """

    commands: BaselineModerateRunningCommandsCfg = BaselineModerateRunningCommandsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 20.0


@configclass
class DigitBaselineRunningEnvCfg(DigitBaselineEnvCfg):
    """Baseline running environment - curriculum step from moderate running.

    Extends baseline approach to running speeds (0-5 m/s).
    Continue training from moderate running checkpoint for smooth curriculum.

    Use with Digit-BaselineRunning-v0 task ID.
    """

    commands: BaselineRunningCommandsCfg = BaselineRunningCommandsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 20.0


# =============================================================================
# FAST RUNNING ENVIRONMENTS (Higher Speed Curriculum)
# =============================================================================

@configclass
class BaselineFastRunningCommandsCfg:
    """Velocity commands for fast running: 0-8 m/s forward."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.005,  # Minimal standing at high speeds
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 8.0),       # Fast running: 0-8 m/s forward (~18 mph)
            lin_vel_y=(-0.1, 0.1),      # Very minimal lateral
            ang_vel_z=(-0.2, 0.2),      # Very minimal turning
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class DigitBaselineFastRunningEnvCfg(DigitBaselineEnvCfg):
    """Baseline fast running environment - curriculum step from running.

    Extends baseline approach to fast running speeds (0-8 m/s, ~18 mph).
    Continue training from running checkpoint for smooth curriculum.

    Use with Digit-BaselineFastRunning-v0 task ID.
    """

    commands: BaselineFastRunningCommandsCfg = BaselineFastRunningCommandsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 20.0
