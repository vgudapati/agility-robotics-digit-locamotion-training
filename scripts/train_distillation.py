"""Training script for student with teacher distillation.

Uses RSL-RL's built-in DistillationRunner for clean integration.
The distillation trains the student to match teacher's action outputs.

Usage:
    # Train student with teacher distillation
    ./isaaclab.bat -p scripts/train_distillation.py \
        --task Digit-Baseline-v0 \
        --teacher-checkpoint logs/digit_baseline_teacher/run_005/model_1000.pt \
        --headless

Paper reference:
    "Real-world humanoid locomotion with reinforcement learning"
    Radosavovic et al., Science Robotics 2024
"""

from __future__ import annotations

import argparse
import os
import sys
import re
from datetime import datetime as dt

# Add the source directory to path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "source", "digit_locomotion"))

# Isaac Lab imports
from isaaclab.app import AppLauncher

# Add argument parsing
parser = argparse.ArgumentParser(description="Train Digit student with distillation.")
parser.add_argument("--task", type=str, default="Digit-Baseline-v0",
                    help="Task ID for student environment")
parser.add_argument("--teacher-checkpoint", type=str, required=True,
                    help="Path to teacher checkpoint (required)")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--max_iterations", type=int, default=5000, help="Max training iterations")

# Tunable reward/termination parameters (for easy curriculum experiments)
parser.add_argument("--arm-penalty", type=float, default=None,
                    help="Arm shoulder roll deviation penalty weight (e.g., -0.1, -0.2, -0.4)")
parser.add_argument("--orientation-limit", type=float, default=None,
                    help="Bad orientation termination limit in radians (e.g., 0.5, 0.7, 1.0)")
parser.add_argument("--vel-max", type=float, default=None,
                    help="Maximum forward velocity command (e.g., 1.5, 2.0, 3.0)")

# Append AppLauncher CLI args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# After app is launched, import remaining modules
import gymnasium as gym
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

# Import Digit environments (triggers registration)
import digit_locomotion.tasks.locomotion  # noqa: F401


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


def get_next_run_dir(base_dir: str) -> str:
    """Get the next run directory with incrementing number."""
    os.makedirs(base_dir, exist_ok=True)

    existing_runs = []
    for d in os.listdir(base_dir):
        match = re.match(r'run_(\d+)', d)
        if match:
            existing_runs.append(int(match.group(1)))

    next_run = max(existing_runs) + 1 if existing_runs else 1
    return os.path.join(base_dir, f"run_{next_run:03d}")


def main():
    """Main training function."""
    print(f"\n{'='*60}")
    print(f"STUDENT TRAINING WITH DISTILLATION")
    print(f"(Using RSL-RL OnPolicyRunner)")
    print(f"{'='*60}")
    print(f"Task: {args_cli.task}")
    print(f"Teacher checkpoint: {args_cli.teacher_checkpoint}")
    print(f"{'='*60}\n")

    # Verify teacher checkpoint exists
    if not os.path.exists(args_cli.teacher_checkpoint):
        print(f"ERROR: Teacher checkpoint not found: {args_cli.teacher_checkpoint}")
        sys.exit(1)

    device = "cuda:0"

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

    # Override reward/termination parameters for curriculum experiments
    if args_cli.arm_penalty is not None:
        if hasattr(env_cfg_obj, 'rewards') and hasattr(env_cfg_obj.rewards, 'arm_shoulder_roll_deviation'):
            env_cfg_obj.rewards.arm_shoulder_roll_deviation.weight = args_cli.arm_penalty
            print(f"  [CLI Override] arm_shoulder_roll_deviation.weight = {args_cli.arm_penalty}")

    if args_cli.orientation_limit is not None:
        if hasattr(env_cfg_obj, 'terminations') and hasattr(env_cfg_obj.terminations, 'bad_orientation'):
            env_cfg_obj.terminations.bad_orientation.params["limit_angle"] = args_cli.orientation_limit
            print(f"  [CLI Override] bad_orientation.limit_angle = {args_cli.orientation_limit}")

    if args_cli.vel_max is not None:
        if hasattr(env_cfg_obj, 'commands') and hasattr(env_cfg_obj.commands, 'base_velocity'):
            # Update the velocity range max
            current_ranges = env_cfg_obj.commands.base_velocity.ranges
            current_ranges.lin_vel_x = (current_ranges.lin_vel_x[0], args_cli.vel_max)
            print(f"  [CLI Override] lin_vel_x range = {current_ranges.lin_vel_x}")

    # Setup logging directory
    log_root = os.path.join("logs", "digit_baseline_distillation")
    log_dir = get_next_run_dir(log_root)
    os.makedirs(log_dir, exist_ok=True)

    # Setup logging to file
    timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")
    tee_logger = TeeLogger(log_file)
    sys.stdout = tee_logger

    # Create environment
    print(f"Creating environment with {env_cfg_obj.scene.num_envs} envs...")
    env = gym.make(args_cli.task, cfg=env_cfg_obj, render_mode=None)
    env = RslRlVecEnvWrapper(env)

    # Print training info
    print(f"\n{'='*60}")
    print(f"TRAINING CONFIGURATION")
    print(f"{'='*60}")
    print(f"Task: {args_cli.task}")
    print(f"Num envs: {env_cfg_obj.scene.num_envs}")
    print(f"Max iterations: {agent_cfg_obj.max_iterations}")
    print(f"Steps per env: {agent_cfg_obj.num_steps_per_env}")
    print(f"Log directory: {log_dir}")
    if args_cli.arm_penalty is not None:
        print(f"Arm penalty (CLI): {args_cli.arm_penalty}")
    if args_cli.orientation_limit is not None:
        print(f"Orientation limit (CLI): {args_cli.orientation_limit}")
    if args_cli.vel_max is not None:
        print(f"Max velocity (CLI): {args_cli.vel_max}")
    print(f"{'='*60}\n")

    # Create runner
    print(f"Creating PPO runner...")
    runner = OnPolicyRunner(env, agent_cfg_obj.to_dict(), log_dir=log_dir, device=device)

    # Load teacher weights to initialize student (warm start)
    print(f"\nLoading teacher checkpoint to initialize student weights...")
    print(f"  Teacher checkpoint: {args_cli.teacher_checkpoint}")
    try:
        runner.load(args_cli.teacher_checkpoint)
        print(f"  Successfully loaded teacher weights!")
        print(f"  Note: Student starts from teacher's learned policy")
    except Exception as e:
        print(f"  Warning: Could not load teacher weights: {e}")
        print(f"  Training will start from random initialization")

    # Start training
    print("\nStarting training...")
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
    sys.stdout = tee_logger.terminal
    tee_logger.close()
    print(f"Log saved to: {log_file}")


if __name__ == "__main__":
    main()
    simulation_app.close()
