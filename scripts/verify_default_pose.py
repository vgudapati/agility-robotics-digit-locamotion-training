"""Script to verify and visualize DIGIT robot default pose.

This script loads the DIGIT robot and displays:
1. Default joint positions for all joints
2. Specifically highlights arm joint positions
3. Allows visual inspection of the pose

Usage:
    C:\IsaacLab\isaaclab.bat -p scripts/verify_default_pose.py
    C:\IsaacLab\isaaclab.bat -p scripts/verify_default_pose.py --headless  # No GUI
"""

import argparse
import torch

from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="Verify DIGIT robot default pose")
parser.add_argument("--headless", action="store_true", help="Run in headless mode")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
args = parser.parse_args()

# Launch app
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Now import Isaac Lab modules
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# DIGIT robot configuration
DIGIT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Agility/Digit/digit_v4.usd",
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.05),
    ),
    actuators={
        "all": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=None,
            damping=None,
        ),
    },
)


def main():
    """Main function to verify default pose."""

    # Create simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device="cuda:0")
    sim = SimulationContext(sim_cfg)

    # Set camera view
    sim.set_camera_view(eye=[3.0, 3.0, 2.0], target=[0.0, 0.0, 1.0])

    # Spawn ground plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)

    # Spawn DIGIT robot
    digit_cfg = DIGIT_CFG.copy()
    digit_cfg.prim_path = "/World/Robot"
    digit = Articulation(cfg=digit_cfg)

    # Play simulation to initialize
    sim.reset()

    # Get joint information
    joint_names = digit.joint_names
    default_joint_pos = digit.data.default_joint_pos[0].cpu().numpy()
    current_joint_pos = digit.data.joint_pos[0].cpu().numpy()

    print("\n" + "=" * 80)
    print("DIGIT ROBOT DEFAULT POSE ANALYSIS")
    print("=" * 80)

    print(f"\nTotal joints: {len(joint_names)}")

    # Categorize joints
    arm_joints = []
    leg_joints = []
    other_joints = []

    for i, name in enumerate(joint_names):
        joint_info = {
            "index": i,
            "name": name,
            "default_pos": default_joint_pos[i],
            "current_pos": current_joint_pos[i],
        }

        if "_arm_" in name.lower():
            arm_joints.append(joint_info)
        elif any(x in name.lower() for x in ["hip", "knee", "toe", "leg"]):
            leg_joints.append(joint_info)
        else:
            other_joints.append(joint_info)

    # Print arm joints
    print("\n" + "-" * 40)
    print("ARM JOINTS (these need to change for hanging arms)")
    print("-" * 40)
    for j in arm_joints:
        print(f"  [{j['index']:2d}] {j['name']:30s} default: {j['default_pos']:8.4f} rad ({j['default_pos']*180/3.14159:8.2f} deg)")

    # Print leg joints
    print("\n" + "-" * 40)
    print("LEG JOINTS")
    print("-" * 40)
    for j in leg_joints:
        print(f"  [{j['index']:2d}] {j['name']:30s} default: {j['default_pos']:8.4f} rad ({j['default_pos']*180/3.14159:8.2f} deg)")

    # Print other joints
    if other_joints:
        print("\n" + "-" * 40)
        print("OTHER JOINTS")
        print("-" * 40)
        for j in other_joints:
            print(f"  [{j['index']:2d}] {j['name']:30s} default: {j['default_pos']:8.4f} rad ({j['default_pos']*180/3.14159:8.2f} deg)")

    print("\n" + "=" * 80)
    print("RECOMMENDATIONS FOR HANGING ARM POSITION")
    print("=" * 80)
    print("""
To make arms hang down naturally, you need to modify the shoulder pitch joints.
Typical changes needed:
- Shoulder pitch: change from 0 to ~1.57 rad (90 deg) to point arms down
- Shoulder roll: may need adjustment depending on robot kinematics
- Elbow: typically keep at 0 or slight bend

The exact values depend on DIGIT's joint conventions.
Look at the arm joint values above and identify which ones control:
1. Shoulder abduction/adduction (sideways movement)
2. Shoulder flexion/extension (forward/backward)
3. Elbow flexion/extension
""")

    # Run simulation loop for visualization
    print("\nSimulation running. Press Ctrl+C to exit.")

    try:
        while simulation_app.is_running():
            sim.step()
            digit.update(sim.cfg.dt)
    except KeyboardInterrupt:
        print("\nExiting...")

    # Cleanup
    simulation_app.close()


if __name__ == "__main__":
    main()
