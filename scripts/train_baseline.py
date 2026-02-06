"""Training script for baseline approach (Radosavovic et al. 2024).

This script implements the two-stage training from the Science Robotics paper:
1. Stage 1: Train teacher policy (MLP with privileged state)
2. Stage 2: Train student policy (Transformer with noisy obs + teacher distillation)

Usage:
    # Stage 1: Train teacher (MLP with privileged state, fast convergence)
    ./isaaclab.bat -p scripts/train_baseline.py --task Digit-BaselineTeacher-v0 --headless

    # Stage 2: Train student with MLP baseline (for comparison)
    ./isaaclab.bat -p scripts/train_baseline.py --task Digit-Baseline-v0 --headless

    # Alternative: Train LSTM baseline (for comparison)
    ./isaaclab.bat -p scripts/train_baseline.py --task Digit-BaselineLSTM-v0 --headless

    # Optional: Resume from checkpoint
    ./isaaclab.bat -p scripts/train_baseline.py --task Digit-Baseline-v0 --resume \
        --checkpoint logs/digit_baseline/run_001/model_5000.pt --headless

Paper reference:
    "Real-world humanoid locomotion with reinforcement learning"
    Radosavovic et al., Science Robotics 2024
"""

from __future__ import annotations

import argparse
import os
import sys
import datetime
import re


class TeeLogger:
    """Tee stdout to both console and file."""

    def __init__(self, log_file: str):
        self.terminal = sys.stdout
        self.log_file = open(log_file, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()

# Add the source directory to path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "source", "digit_locomotion"))

# Isaac Lab imports
from isaaclab.app import AppLauncher

# Add argument parsing
parser = argparse.ArgumentParser(description="Train Digit baseline policy.")
parser.add_argument("--task", type=str, default="Digit-Baseline-v0",
                    help="Task ID (Digit-Baseline-v0, Digit-BaselineTeacher-v0, Digit-BaselineLSTM-v0)")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments")
parser.add_argument("--seed", type=int, default=None, help="Random seed")
parser.add_argument("--max_iterations", type=int, default=None, help="Max training iterations")
parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path to resume from")
parser.add_argument("--video", action="store_true", help="Record video during training")
parser.add_argument("--video_length", type=int, default=200, help="Video length in steps")
parser.add_argument("--video_interval", type=int, default=2000, help="Video recording interval")

# Append AppLauncher CLI args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# After app is launched, import remaining modules
import gymnasium as gym
import torch
from datetime import datetime as dt

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

# Import Digit environments (triggers registration)
import digit_locomotion.tasks.locomotion  # noqa: F401

# Import custom transformer policy for RSL-RL integration
from digit_locomotion.networks.transformer_policy import ActorCriticTransformer  # noqa: F401


def get_next_run_dir(base_dir: str) -> str:
    """Get the next run directory with incrementing number."""
    os.makedirs(base_dir, exist_ok=True)

    # Find existing run directories
    existing_runs = []
    for d in os.listdir(base_dir):
        match = re.match(r'run_(\d+)', d)
        if match:
            existing_runs.append(int(match.group(1)))

    # Get next run number
    next_run = max(existing_runs) + 1 if existing_runs else 1

    return os.path.join(base_dir, f"run_{next_run:03d}")


def find_latest_checkpoint(log_dir: str) -> str | None:
    """Find the latest checkpoint in the log directory."""
    if not os.path.exists(log_dir):
        return None

    checkpoints = []
    for f in os.listdir(log_dir):
        match = re.match(r'model_(\d+)\.pt', f)
        if match:
            checkpoints.append((int(match.group(1)), os.path.join(log_dir, f)))

    if not checkpoints:
        return None

    # Return checkpoint with highest iteration number
    checkpoints.sort(key=lambda x: x[0], reverse=True)
    return checkpoints[0][1]


def main():
    """Main training function."""
    print(f"\n{'='*60}")
    print(f"BASELINE TRAINING (Radosavovic et al. 2024)")
    print(f"{'='*60}")
    print(f"Task: {args_cli.task}")
    print(f"Resume: {args_cli.resume}")
    print(f"{'='*60}\n")

    # Get environment spec
    env_cfg = gym.spec(args_cli.task).kwargs["env_cfg_entry_point"]
    agent_cfg = gym.spec(args_cli.task).kwargs["rsl_rl_cfg_entry_point"]

    # Parse configs
    env_cfg_class = env_cfg.rsplit(":", 1)
    env_cfg_module = __import__(env_cfg_class[0], fromlist=[env_cfg_class[1]])
    env_cfg_obj = getattr(env_cfg_module, env_cfg_class[1])()

    agent_cfg_class = agent_cfg.rsplit(":", 1)
    agent_cfg_module = __import__(agent_cfg_class[0], fromlist=[agent_cfg_class[1]])
    agent_cfg_obj = getattr(agent_cfg_module, agent_cfg_class[1])()

    # Override with CLI args
    if args_cli.num_envs is not None:
        env_cfg_obj.scene.num_envs = args_cli.num_envs
    if args_cli.seed is not None:
        agent_cfg_obj.seed = args_cli.seed
    if args_cli.max_iterations is not None:
        agent_cfg_obj.max_iterations = args_cli.max_iterations

    # Setup logging directory
    log_root = os.path.join("logs", agent_cfg_obj.experiment_name)

    if args_cli.resume and args_cli.checkpoint:
        # Use parent directory of checkpoint
        log_dir = os.path.dirname(args_cli.checkpoint)
    else:
        # Create new run directory
        log_dir = get_next_run_dir(log_root)

    os.makedirs(log_dir, exist_ok=True)

    # Setup logging to file
    timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")
    tee_logger = TeeLogger(log_file)
    sys.stdout = tee_logger

    # Set resume flag
    agent_cfg_obj.resume = args_cli.resume

    # Create environment
    print(f"Creating environment with {env_cfg_obj.scene.num_envs} envs...")
    env = gym.make(args_cli.task, cfg=env_cfg_obj, render_mode=None)

    # Wrap environment for RSL-RL
    env = RslRlVecEnvWrapper(env)

    # Create runner
    print(f"Creating PPO runner...")
    print(f"  Policy type: {agent_cfg_obj.policy.__class__.__name__}")
    print(f"  Log directory: {log_dir}")

    runner = OnPolicyRunner(env, agent_cfg_obj.to_dict(), log_dir=log_dir, device="cuda:0")

    # Load checkpoint if resuming
    if args_cli.resume:
        checkpoint_path = args_cli.checkpoint
        if checkpoint_path is None:
            checkpoint_path = find_latest_checkpoint(log_dir)

        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Loading checkpoint: {checkpoint_path}")
            runner.load(checkpoint_path)
        else:
            print(f"Warning: No checkpoint found, starting from scratch")

    # Print training info
    print(f"\n{'='*60}")
    print(f"TRAINING CONFIGURATION")
    print(f"{'='*60}")
    print(f"Task: {args_cli.task}")
    print(f"Num envs: {env_cfg_obj.scene.num_envs}")
    print(f"Max iterations: {agent_cfg_obj.max_iterations}")
    print(f"Steps per env: {agent_cfg_obj.num_steps_per_env}")
    print(f"Log directory: {log_dir}")
    print(f"{'='*60}\n")

    # Start training
    print("Starting training...")
    start_time = dt.now()

    runner.learn(num_learning_iterations=agent_cfg_obj.max_iterations, init_at_random_ep_len=True)

    # Save final model
    final_model_path = os.path.join(log_dir, "model_final.pt")
    runner.save(final_model_path)

    end_time = dt.now()
    duration = end_time - start_time

    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Duration: {duration}")
    print(f"Final model: {final_model_path}")
    print(f"{'='*60}\n")

    # Cleanup
    env.close()

    # Close log file
    sys.stdout = tee_logger.terminal
    tee_logger.close()
    print(f"Log saved to: {log_file}")


if __name__ == "__main__":
    main()
    simulation_app.close()
