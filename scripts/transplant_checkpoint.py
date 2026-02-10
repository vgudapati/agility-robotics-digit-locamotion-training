"""Transplant checkpoint weights from full action space to legs-only action space.

Removes arm joint output neurons from the actor's final layer while preserving
all hidden layer weights (leg gait knowledge). The critic is reset since its
value estimates are no longer valid with a different action space.

Usage:
    # First, run print_joint_names.py to identify arm indices
    # Then run this script:
    C:\\IsaacLab\\_isaac_sim\\python.bat scripts/transplant_checkpoint.py \\
        --checkpoint C:\\IsaacLab\\logs\\digit_baseline_teacher\\run_042\\model_3350.pt \\
        --output C:\\IsaacLab\\logs\\digit_baseline_teacher\\model_3350_legs_only.pt

    # Or auto-detect from environment (requires IsaacSim):
    C:\\IsaacLab\\_isaac_sim\\python.bat scripts/transplant_checkpoint.py \\
        --checkpoint C:\\IsaacLab\\logs\\digit_baseline_teacher\\run_042\\model_3350.pt \\
        --output C:\\IsaacLab\\logs\\digit_baseline_teacher\\model_3350_legs_only.pt \\
        --auto-detect
"""
import argparse
import re
import sys
import torch


# Known Digit V4 arm joint name pattern (from USD: left_arm_shoulder_pitch, etc.)
# Matches all 14 arm joints: shoulder (roll/pitch/yaw), elbow, wrist (roll/pitch/yaw) x2
ARM_PATTERN = re.compile(r".*_arm_.*")


def is_arm_joint(name: str) -> bool:
    """Check if a joint name matches arm pattern (.*_arm_.*)."""
    return bool(ARM_PATTERN.fullmatch(name))


def get_joint_names_from_env():
    """Load the environment and get actual joint names (requires IsaacSim)."""
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args([])
    args.headless = True
    args.num_envs = 1
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import isaaclab.sim as sim_utils
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.assets import AssetBaseCfg
    from isaaclab_assets.robots.agility import DIGIT_V4_CFG

    scene_cfg = InteractiveSceneCfg(num_envs=1, env_spacing=2.0)
    scene_cfg.robot = DIGIT_V4_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    scene_cfg.ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    sim_cfg = sim_utils.SimulationCfg(dt=0.005)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])

    scene = InteractiveScene(scene_cfg)
    sim.reset()
    scene.reset()

    joint_names = list(scene["robot"].joint_names)
    simulation_app.close()
    return joint_names


def transplant(checkpoint_path: str, output_path: str, joint_names: list[str] | None = None):
    """Transplant checkpoint from full action space to legs-only."""
    print(f"Loading checkpoint: {checkpoint_path}")
    loaded = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    model_state = loaded["model_state_dict"]

    # Detect action space size from actor output layer
    actor_output_weight_key = None
    actor_output_bias_key = None
    for key in sorted(model_state.keys()):
        if "actor" in key and "weight" in key:
            actor_output_weight_key = key
        if "actor" in key and "bias" in key:
            actor_output_bias_key = key
    # Last actor weight/bias are the output layer
    old_num_actions = model_state[actor_output_bias_key].shape[0]
    print(f"Original action space size: {old_num_actions}")

    if joint_names is not None:
        # Use actual joint names to determine indices
        assert len(joint_names) == old_num_actions, (
            f"Joint names count ({len(joint_names)}) != action space size ({old_num_actions})"
        )
        arm_indices = [i for i, name in enumerate(joint_names) if is_arm_joint(name)]
        leg_indices = [i for i, name in enumerate(joint_names) if not is_arm_joint(name)]
        print(f"\nJoint mapping from environment:")
        for i, name in enumerate(joint_names):
            marker = " <-- ARM (removing)" if is_arm_joint(name) else ""
            print(f"  [{i:2d}] {name}{marker}")
    else:
        # Fallback: assume arm joints are the last 8 (common Digit ordering)
        # This is based on the joint ordering in digit.py:
        #   indices 0-15: leg joints (left leg 0-7, right leg 8-15)
        #   indices 16-23: arm joints (left arm 16-19, right arm 20-23)
        # But the USD may have more joints (50 total), so this is approximate.
        print("\nWARNING: No joint names provided. Cannot determine exact arm indices.")
        print("Run with --auto-detect or provide joint names via print_joint_names.py first.")
        print("Aborting to prevent incorrect transplant.")
        sys.exit(1)

    new_num_actions = len(leg_indices)
    print(f"\nArm joints to remove: {len(arm_indices)} (indices: {arm_indices})")
    print(f"New action space size: {new_num_actions}")

    # The observation space also changes because `actions` (last_action) shrinks
    # from 50 -> 36. Observation layout:
    #   [commands(3), lin_vel(3), ang_vel(3), gravity(3), joint_pos(50), joint_vel(50), actions(50)]
    #   Total old: 162, new: 148 (actions goes from 50 -> 36)
    # We need to trim the action columns from the first layer of actor and critic.
    obs_prefix_size = 3 + 3 + 3 + 3 + 50 + 50  # = 112 (everything before actions)
    old_obs_size = obs_prefix_size + old_num_actions  # 112 + 50 = 162
    new_obs_size = obs_prefix_size + new_num_actions  # 112 + 36 = 148

    # Columns to keep: all prefix columns + leg action columns
    keep_columns = list(range(obs_prefix_size)) + [obs_prefix_size + i for i in leg_indices]
    keep_columns_tensor = torch.tensor(keep_columns, dtype=torch.long)
    print(f"\nObservation space: {old_obs_size} -> {new_obs_size}")

    # Create new model state dict
    new_model_state = {}
    leg_indices_tensor = torch.tensor(leg_indices, dtype=torch.long)

    # Find first layer keys for actor and critic (input layers that need column trimming)
    actor_input_weight_key = sorted([k for k in model_state if "actor" in k and "weight" in k])[0]
    critic_input_weight_key = sorted([k for k in model_state if "critic" in k and "weight" in k])[0]
    actor_input_bias_key = sorted([k for k in model_state if "actor" in k and "bias" in k])[0]
    critic_input_bias_key = sorted([k for k in model_state if "critic" in k and "bias" in k])[0]

    for key, tensor in model_state.items():
        if key == actor_output_weight_key:
            # Output weight: shape [num_actions, hidden_dim] -> keep only leg rows
            new_model_state[key] = tensor[leg_indices_tensor]
            print(f"  {key}: {tensor.shape} -> {new_model_state[key].shape}")
        elif key == actor_output_bias_key:
            # Output bias: shape [num_actions] -> keep only leg entries
            new_model_state[key] = tensor[leg_indices_tensor]
            print(f"  {key}: {tensor.shape} -> {new_model_state[key].shape}")
        elif key == "std":
            # Action noise std: shape [num_actions] -> keep only leg entries
            new_model_state[key] = tensor[leg_indices_tensor]
            print(f"  {key}: {tensor.shape} -> {new_model_state[key].shape}")
        elif key in (actor_input_weight_key, critic_input_weight_key):
            # Input weight: shape [hidden_dim, obs_size] -> trim action columns
            new_model_state[key] = tensor[:, keep_columns_tensor]
            print(f"  {key}: {tensor.shape} -> {new_model_state[key].shape}")
        else:
            # All other layers: keep as-is (hidden layers, biases, critic output)
            new_model_state[key] = tensor

    # Build new checkpoint
    new_checkpoint = {
        "model_state_dict": new_model_state,
        "optimizer_state_dict": {},  # Reset optimizer (Adam state invalid for new architecture)
        "iter": loaded["iter"],
        "infos": loaded.get("infos"),
    }

    torch.save(new_checkpoint, output_path)
    print(f"\nSaved transplanted checkpoint to: {output_path}")
    print(f"  Actions: {old_num_actions} -> {new_num_actions}")
    print(f"  Optimizer: RESET (Adam state invalid for changed output layer)")
    print(f"  Iteration: {loaded['iter']}")
    print(f"\nNote: Use --learning_rate with a fresh value since optimizer state is reset.")


def main():
    parser = argparse.ArgumentParser(description="Transplant checkpoint to legs-only action space")
    parser.add_argument("--checkpoint", required=True, help="Path to source checkpoint")
    parser.add_argument("--output", required=True, help="Path for transplanted checkpoint")
    parser.add_argument("--digit-v4", action="store_true",
                        help="Use verified Digit V4 joint names (no IsaacSim needed)")
    parser.add_argument("--auto-detect", action="store_true",
                        help="Auto-detect joint names from environment (requires IsaacSim, slow)")
    parser.add_argument("--joint-names-file", type=str, default=None,
                        help="File with one joint name per line (from print_joint_names.py)")
    args = parser.parse_args()

    joint_names = None
    if args.digit_v4:
        # Verified joint names from print_joint_names.py (Digit V4 USD)
        joint_names = [
            "left_leg_hip_roll",           # 0
            "left_arm_shoulder_pitch",     # 1  ARM
            "right_leg_hip_roll",          # 2
            "right_arm_shoulder_pitch",    # 3  ARM
            "left_leg_hip_yaw",            # 4
            "left_arm_shoulder_roll",      # 5  ARM
            "right_leg_hip_yaw",           # 6
            "right_arm_shoulder_roll",     # 7  ARM
            "left_leg_hip_pitch",          # 8
            "left_arm_shoulder_yaw",       # 9  ARM
            "right_leg_hip_pitch",         # 10
            "right_arm_shoulder_yaw",      # 11 ARM
            "left_leg_knee",               # 12
            "left_leg_achilles_rod:0",     # 13
            "left_leg_achilles_rod:1",     # 14
            "left_leg_achilles_rod:2",     # 15
            "left_arm_elbow",              # 16 ARM
            "right_leg_achilles_rod:0",    # 17
            "right_leg_achilles_rod:1",    # 18
            "right_leg_achilles_rod:2",    # 19
            "right_leg_knee",              # 20
            "right_arm_elbow",             # 21 ARM
            "left_leg_tarsus",             # 22
            "left_arm_wrist_roll",         # 23 ARM
            "right_leg_tarsus",            # 24
            "right_arm_wrist_roll",        # 25 ARM
            "left_leg_toe_a",              # 26
            "left_leg_toe_b",              # 27
            "left_leg_toe_pitch",          # 28
            "left_arm_wrist_pitch",        # 29 ARM
            "right_leg_toe_a",             # 30
            "right_leg_toe_b",             # 31
            "right_leg_toe_pitch",         # 32
            "right_arm_wrist_pitch",       # 33 ARM
            "left_leg_toe_a_rod:0",        # 34
            "left_leg_toe_a_rod:1",        # 35
            "left_leg_toe_a_rod:2",        # 36
            "left_leg_toe_b_rod:0",        # 37
            "left_leg_toe_b_rod:1",        # 38
            "left_leg_toe_b_rod:2",        # 39
            "left_leg_toe_roll",           # 40
            "left_arm_wrist_yaw",          # 41 ARM
            "right_leg_toe_a_rod:0",       # 42
            "right_leg_toe_a_rod:1",       # 43
            "right_leg_toe_a_rod:2",       # 44
            "right_leg_toe_b_rod:0",       # 45
            "right_leg_toe_b_rod:1",       # 46
            "right_leg_toe_b_rod:2",       # 47
            "right_leg_toe_roll",          # 48
            "right_arm_wrist_yaw",         # 49 ARM
        ]
        print(f"Using verified Digit V4 joint names ({len(joint_names)} joints)")
    elif args.auto_detect:
        print("Auto-detecting joint names from environment (this may take a while)...")
        joint_names = get_joint_names_from_env()
    elif args.joint_names_file:
        with open(args.joint_names_file) as f:
            joint_names = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(joint_names)} joint names from {args.joint_names_file}")

    transplant(args.checkpoint, args.output, joint_names)


if __name__ == "__main__":
    main()
