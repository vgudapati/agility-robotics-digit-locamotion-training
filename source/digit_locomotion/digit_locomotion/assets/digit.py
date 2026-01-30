"""Configuration for Agility Robotics Digit humanoid robot.

Digit is a bipedal humanoid robot with approximately 30 degrees of freedom:
- 2x Legs: Each with hip (roll/yaw/pitch), knee, shin, tarsus, toe (pitch/roll)
- 2x Arms: Each with shoulder (roll/pitch/yaw), elbow
- Unique 4-bar linkage design for the legs

This configuration uses the built-in Digit USD asset from Isaac Sim.
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg, DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg


# Joint names for Digit robot (these should match the USD asset)
# Note: Actual joint names may vary based on the specific Digit model version
DIGIT_LEG_JOINT_NAMES = [
    "left_hip_roll",
    "left_hip_yaw",
    "left_hip_pitch",
    "left_knee",
    "left_shin",
    "left_tarsus",
    "left_toe_pitch",
    "left_toe_roll",
    "right_hip_roll",
    "right_hip_yaw",
    "right_hip_pitch",
    "right_knee",
    "right_shin",
    "right_tarsus",
    "right_toe_pitch",
    "right_toe_roll",
]

DIGIT_ARM_JOINT_NAMES = [
    "left_shoulder_roll",
    "left_shoulder_pitch",
    "left_shoulder_yaw",
    "left_elbow",
    "right_shoulder_roll",
    "right_shoulder_pitch",
    "right_shoulder_yaw",
    "right_elbow",
]

# Default standing pose for Digit
# These values provide a stable initial configuration
DIGIT_DEFAULT_JOINT_POS = {
    # Left leg - slight knee bend for stability
    "left_hip_roll": 0.0,
    "left_hip_yaw": 0.0,
    "left_hip_pitch": -0.2,
    "left_knee": 0.4,
    "left_shin": 0.0,
    "left_tarsus": -0.2,
    "left_toe_pitch": 0.0,
    "left_toe_roll": 0.0,
    # Right leg - mirror of left
    "right_hip_roll": 0.0,
    "right_hip_yaw": 0.0,
    "right_hip_pitch": -0.2,
    "right_knee": 0.4,
    "right_shin": 0.0,
    "right_tarsus": -0.2,
    "right_toe_pitch": 0.0,
    "right_toe_roll": 0.0,
    # Left arm - slightly forward and bent
    "left_shoulder_roll": 0.0,
    "left_shoulder_pitch": 0.3,
    "left_shoulder_yaw": 0.0,
    "left_elbow": -0.3,
    # Right arm - mirror of left
    "right_shoulder_roll": 0.0,
    "right_shoulder_pitch": 0.3,
    "right_shoulder_yaw": 0.0,
    "right_elbow": -0.3,
}


DIGIT_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        # Use the built-in Digit asset from Isaac Sim
        # If you have a custom URDF, convert it using:
        # ./isaaclab.sh -p scripts/tools/convert_urdf.py <urdf_path> <output_usd_path>
        usd_path="${ISAACLAB_NUCLEUS_DIR}/Robots/Agility/Digit/digit_v4.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # Digit stands approximately 1.0m tall at the hip
        pos=(0.0, 0.0, 1.05),
        rot=(1.0, 0.0, 0.0, 0.0),  # wxyz quaternion
        joint_pos=DIGIT_DEFAULT_JOINT_POS,
        joint_vel={".*": 0.0},
    ),
    actuators={
        # Leg actuators - high torque for locomotion
        # Using IdealPD for stable training, can switch to DCMotor for more realism
        "legs": IdealPDActuatorCfg(
            joint_names_expr=[
                ".*hip_roll",
                ".*hip_yaw",
                ".*hip_pitch",
                ".*knee",
                ".*shin",
                ".*tarsus",
                ".*toe_pitch",
                ".*toe_roll",
            ],
            effort_limit=150.0,  # Nm - adjust based on actual Digit specs
            velocity_limit=10.0,  # rad/s
            stiffness=80.0,
            damping=4.0,
        ),
        # Arm actuators - lower torque requirements
        "arms": IdealPDActuatorCfg(
            joint_names_expr=[
                ".*shoulder_roll",
                ".*shoulder_pitch",
                ".*shoulder_yaw",
                ".*elbow",
            ],
            effort_limit=40.0,  # Nm
            velocity_limit=8.0,  # rad/s
            stiffness=40.0,
            damping=2.0,
        ),
    },
    soft_joint_pos_limit_factor=0.95,
)


# Minimal configuration for faster simulation (legs only, arms fixed)
DIGIT_MINIMAL_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path="${ISAACLAB_NUCLEUS_DIR}/Robots/Agility/Digit/digit_v4.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.05),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos=DIGIT_DEFAULT_JOINT_POS,
        joint_vel={".*": 0.0},
    ),
    actuators={
        # Only leg actuators for simplified training
        "legs": IdealPDActuatorCfg(
            joint_names_expr=[
                ".*hip_roll",
                ".*hip_yaw",
                ".*hip_pitch",
                ".*knee",
                ".*shin",
                ".*tarsus",
                ".*toe_pitch",
                ".*toe_roll",
            ],
            effort_limit=150.0,
            velocity_limit=10.0,
            stiffness=80.0,
            damping=4.0,
        ),
        # Arms held in fixed position with high stiffness
        "arms_fixed": IdealPDActuatorCfg(
            joint_names_expr=[
                ".*shoulder_roll",
                ".*shoulder_pitch",
                ".*shoulder_yaw",
                ".*elbow",
            ],
            effort_limit=100.0,
            velocity_limit=0.0,  # No movement
            stiffness=200.0,  # High stiffness to hold position
            damping=20.0,
        ),
    },
    soft_joint_pos_limit_factor=0.95,
)
