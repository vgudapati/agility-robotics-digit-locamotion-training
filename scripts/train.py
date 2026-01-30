#!/usr/bin/env python3
"""Training script for Digit locomotion policy.

This script trains a locomotion policy for the Agility Robotics Digit
robot using Isaac Lab and RSL-RL's PPO implementation.

Usage (Linux/macOS):
    # Train on flat terrain (recommended for initial training)
    ./isaaclab.sh -p scripts/train.py --task Digit-Velocity-Flat-v0

    # Train on rough terrain
    ./isaaclab.sh -p scripts/train.py --task Digit-Velocity-Rough-v0

    # Train with custom settings
    ./isaaclab.sh -p scripts/train.py --task Digit-Velocity-Flat-v0 \
        --num_envs 4096 --max_iterations 20000 --headless

Usage (Windows):
    # Train on flat terrain
    isaaclab.bat -p scripts\\train.py --task Digit-Velocity-Flat-v0

    # Train on rough terrain
    isaaclab.bat -p scripts\\train.py --task Digit-Velocity-Rough-v0

    # Train with custom settings
    isaaclab.bat -p scripts\\train.py --task Digit-Velocity-Flat-v0 ^
        --num_envs 4096 --max_iterations 20000 --headless

    # Resume training from checkpoint
    isaaclab.bat -p scripts\\train.py --task Digit-Velocity-Flat-v0 ^
        --resume --load_run <run_name>
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

# === Parse Arguments (before Isaac imports) ===

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a locomotion policy for Digit robot"
    )

    # Environment settings
    parser.add_argument(
        "--task",
        type=str,
        default="Digit-Velocity-Flat-v0",
        help="Name of the task/environment to train",
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=None,
        help="Number of parallel environments (default: from config)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    # Training settings
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=None,
        help="Maximum training iterations (default: from config)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from checkpoint",
    )
    parser.add_argument(
        "--load_run",
        type=str,
        default=None,
        help="Name of the run to load when resuming",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint file to load",
    )

    # Logging settings
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help="Directory to save logs and checkpoints",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Name of the experiment (default: from config)",
    )

    # Add Isaac Sim launcher arguments
    AppLauncher.add_app_launcher_args(parser)

    return parser.parse_args()


# Parse arguments before importing Isaac modules
args = parse_args()

# Launch the Isaac Sim application
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


# === Now import Isaac modules (after app launch) ===

import gymnasium as gym
import torch

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

# Import to register environments
import digit_locomotion  # noqa: F401


def main():
    """Main training function."""
    # Set random seed
    torch.manual_seed(args.seed)

    # Create the environment
    env_cfg_kwargs = {}
    if args.num_envs is not None:
        env_cfg_kwargs["num_envs"] = args.num_envs

    env = gym.make(
        args.task,
        cfg=env_cfg_kwargs if env_cfg_kwargs else None,
    )

    # Wrap environment for RSL-RL
    env = RslRlVecEnvWrapper(env)

    # Get agent configuration
    agent_cfg: RslRlOnPolicyRunnerCfg = env.unwrapped.cfg.agent_cfg

    # Override configuration with command line arguments
    if args.max_iterations is not None:
        agent_cfg.max_iterations = args.max_iterations
    if args.experiment_name is not None:
        agent_cfg.experiment_name = args.experiment_name

    # Set resume parameters
    agent_cfg.resume = args.resume
    if args.load_run is not None:
        agent_cfg.load_run = args.load_run
    if args.checkpoint is not None:
        agent_cfg.load_checkpoint = args.checkpoint

    # Create the log directory
    log_dir = os.path.join(args.log_dir, agent_cfg.experiment_name)
    os.makedirs(log_dir, exist_ok=True)

    # Import RSL-RL runner
    from rsl_rl.runners import OnPolicyRunner

    # Create the runner
    runner = OnPolicyRunner(
        env=env,
        train_cfg=agent_cfg,
        log_dir=log_dir,
        device=env.device,
    )

    # Print training info
    print("\n" + "=" * 60)
    print("DIGIT LOCOMOTION TRAINING")
    print("=" * 60)
    print(f"Task: {args.task}")
    print(f"Number of environments: {env.num_envs}")
    print(f"Max iterations: {agent_cfg.max_iterations}")
    print(f"Log directory: {log_dir}")
    print(f"Device: {env.device}")
    print("=" * 60 + "\n")

    # Run training
    runner.learn(
        num_learning_iterations=agent_cfg.max_iterations,
        init_at_random_ep_len=True,
    )

    # Save final model
    print("\nTraining complete! Saving final model...")
    runner.save(os.path.join(log_dir, "final_model.pt"))

    # Cleanup
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
