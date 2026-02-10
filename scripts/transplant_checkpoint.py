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


# Known Digit V4 arm joint name patterns (from digit.py asset definition)
ARM_PATTERNS = [
    re.compile(r".*shoulder_roll"),
    re.compile(r".*shoulder_pitch"),
    re.compile(r".*shoulder_yaw"),
    re.compile(r".*elbow"),
]


def is_arm_joint(name: str) -> bool:
    """Check if a joint name matches arm patterns."""
    return any(p.fullmatch(name) for p in ARM_PATTERNS)


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

    # Create new model state dict
    new_model_state = {}
    leg_indices_tensor = torch.tensor(leg_indices, dtype=torch.long)

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
        else:
            # All hidden layers and critic layers: keep as-is
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
    parser.add_argument("--auto-detect", action="store_true",
                        help="Auto-detect joint names from environment (requires IsaacSim)")
    parser.add_argument("--joint-names-file", type=str, default=None,
                        help="File with one joint name per line (from print_joint_names.py)")
    args = parser.parse_args()

    joint_names = None
    if args.auto_detect:
        print("Auto-detecting joint names from environment...")
        joint_names = get_joint_names_from_env()
    elif args.joint_names_file:
        with open(args.joint_names_file) as f:
            joint_names = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(joint_names)} joint names from {args.joint_names_file}")

    transplant(args.checkpoint, args.output, joint_names)


if __name__ == "__main__":
    main()
