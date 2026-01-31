#!/usr/bin/env python3
"""Evaluation/playback script for trained Digit locomotion policies.

This script loads a trained policy and runs it in the simulation
environment for visualization and evaluation.

Usage (Windows PowerShell):
    # IMPORTANT: In PowerShell, use .\\ prefix and deactivate conda first
    cd C:\\IsaacLab
    $env:CONDA_PREFIX = ""

    # Play with visualization (4 robots)
    .\\isaaclab.bat -p c:\\path\\to\\scripts\\play.py --num_envs 4

    # Play with FAST forward walking (1.5 m/s)
    .\\isaaclab.bat -p c:\\path\\to\\scripts\\play.py --num_envs 4 --vel_x 1.5

    # Play with custom velocity (forward + turning)
    .\\isaaclab.bat -p c:\\path\\to\\scripts\\play.py --vel_x 1.0 --vel_yaw 0.5

    # Headless evaluation (faster)
    .\\isaaclab.bat -p c:\\path\\to\\scripts\\play.py --headless --num_envs 64

Usage (Windows Command Prompt):
    cd C:\\IsaacLab
    set CONDA_PREFIX=
    isaaclab.bat -p c:\\path\\to\\scripts\\play.py --num_envs 4 --vel_x 1.5

Usage (Linux):
    cd /path/to/IsaacLab
    ./isaaclab.sh -p /path/to/scripts/play.py --num_envs 4 --vel_x 1.5
"""

from __future__ import annotations

import argparse
import os
import sys

# Add the source directory to path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "source", "digit_locomotion"))

from isaaclab.app import AppLauncher


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Digit locomotion policy"
    )

    # Environment settings
    parser.add_argument(
        "--task",
        type=str,
        default="Digit-Velocity-Flat-v0",
        help="Name of the task/environment",
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=16,
        help="Number of parallel environments for evaluation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    # Checkpoint settings
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="C:/IsaacLab/logs/digit_flat/final_model.pt",
        help="Path to the trained policy checkpoint",
    )

    # Velocity command overrides (for testing without modifying training config)
    parser.add_argument(
        "--vel_x",
        type=float,
        default=None,
        help="Fixed forward velocity (m/s). Use 1.0-1.5 for fast walking.",
    )
    parser.add_argument(
        "--vel_y",
        type=float,
        default=None,
        help="Fixed lateral velocity (m/s). Positive = left.",
    )
    parser.add_argument(
        "--vel_yaw",
        type=float,
        default=None,
        help="Fixed yaw rate (rad/s). Positive = turn left.",
    )

    # Add Isaac Sim launcher arguments
    AppLauncher.add_app_launcher_args(parser)

    return parser.parse_args()


args = parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


# === Isaac imports after app launch ===

import gymnasium as gym
import torch

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg, load_cfg_from_registry

# Import to register environments
import digit_locomotion  # noqa: F401


def main():
    """Main evaluation function."""
    # Set random seed
    torch.manual_seed(args.seed)

    # Parse environment configuration from registry
    env_cfg = parse_env_cfg(
        args.task,
        device=args.device,
        num_envs=args.num_envs,
    )

    # Create the environment with parsed config
    env = gym.make(args.task, cfg=env_cfg)

    # Wrap environment for RSL-RL
    env = RslRlVecEnvWrapper(env)

    # Load agent configuration from registry
    agent_cfg: RslRlOnPolicyRunnerCfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")

    # Import RSL-RL runner
    from rsl_rl.runners import OnPolicyRunner

    # Create a temporary log directory
    log_dir = os.path.join(os.path.dirname(args.checkpoint), "play_logs")
    os.makedirs(log_dir, exist_ok=True)

    # Create the runner
    runner = OnPolicyRunner(
        env=env,
        train_cfg=agent_cfg.to_dict(),
        log_dir=log_dir,
        device=env.device,
    )

    # Load the checkpoint
    print(f"\nLoading policy from: {args.checkpoint}")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    runner.load(args.checkpoint)
    print("Checkpoint loaded successfully!")

    # Get the inference policy
    policy = runner.get_inference_policy(device=env.device)

    # Check if user specified fixed velocities
    use_fixed_vel = args.vel_x is not None or args.vel_y is not None or args.vel_yaw is not None
    fixed_vel_x = args.vel_x if args.vel_x is not None else 0.0
    fixed_vel_y = args.vel_y if args.vel_y is not None else 0.0
    fixed_vel_yaw = args.vel_yaw if args.vel_yaw is not None else 0.0

    # Print info
    print("\n" + "=" * 60)
    print("DIGIT LOCOMOTION EVALUATION")
    print("=" * 60)
    print(f"Task: {args.task}")
    print(f"Number of environments: {env.num_envs}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {env.device}")
    if use_fixed_vel:
        print(f"Fixed velocity: x={fixed_vel_x:.2f} m/s, y={fixed_vel_y:.2f} m/s, yaw={fixed_vel_yaw:.2f} rad/s")
    else:
        print("Velocity: Random commands from environment")
    print("=" * 60)
    print("\nRunning policy... Press Ctrl+C to stop\n")

    # Get initial observations
    obs = env.get_observations()

    # Evaluation loop
    step = 0
    episode_rewards = torch.zeros(env.num_envs, device=env.device)
    episode_lengths = torch.zeros(env.num_envs, device=env.device)

    try:
        while simulation_app.is_running():
            # Override velocity commands if specified
            if use_fixed_vel:
                # Access the underlying Isaac Lab environment
                base_env = env.unwrapped
                cmd_manager = base_env.command_manager
                # Set fixed velocity commands for all environments
                cmd_manager.get_command("base_velocity")[:, 0] = fixed_vel_x  # x velocity
                cmd_manager.get_command("base_velocity")[:, 1] = fixed_vel_y  # y velocity
                cmd_manager.get_command("base_velocity")[:, 2] = fixed_vel_yaw  # yaw rate

            # Get action from policy
            with torch.no_grad():
                actions = policy(obs)

            # Step environment
            obs, rewards, dones, infos = env.step(actions)

            # Track metrics
            episode_rewards += rewards
            episode_lengths += 1

            # Print stats periodically
            if step % 50 == 0:
                print(f"Step {step}: Mean reward = {rewards.mean().item():.4f}")

            # Print stats on episode end
            done_indices = dones.nonzero(as_tuple=False).squeeze(-1)
            if len(done_indices) > 0:
                for idx in done_indices:
                    idx = idx.item()
                    print(f"  Episode finished (env {idx}): "
                          f"reward = {episode_rewards[idx].item():.2f}, "
                          f"length = {episode_lengths[idx].item():.0f}")
                # Reset tracked metrics for completed episodes
                episode_rewards[done_indices] = 0
                episode_lengths[done_indices] = 0

            step += 1

    except KeyboardInterrupt:
        print("\n\nEvaluation stopped by user")

    # Cleanup
    print(f"\nTotal steps: {step}")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
