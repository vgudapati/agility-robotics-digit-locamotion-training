"""Training script for baseline approach (Radosavovic et al. 2024).

This script implements the two-stage training from the Science Robotics paper:
1. Stage 1: Train teacher policy (MLP with privileged state)
2. Stage 2: Train student policy (Transformer with noisy obs + teacher distillation)

Usage:
    # Stage 1: Train teacher (single GPU)
    ./isaaclab.sh -p scripts/train_baseline.py --task Digit-BaselineTeacher-v0 --headless

    # Stage 1: Train teacher (distributed, 8 GPUs)
    python -m torch.distributed.run --nnodes=1 --nproc_per_node=8 \
        scripts/train_baseline.py --task Digit-BaselineTeacher-v0 --headless --distributed

    # Stage 1: Train teacher (distributed, 8 GPUs, max envs)
    python -m torch.distributed.run --nnodes=1 --nproc_per_node=8 \
        scripts/train_baseline.py --task Digit-BaselineTeacher-v0 --num_envs 262144 \
        --headless --distributed

    # Stage 2: Train student with MLP baseline (for comparison)
    ./isaaclab.sh -p scripts/train_baseline.py --task Digit-Baseline-v0 --headless

    # Alternative: Train LSTM baseline (for comparison)
    ./isaaclab.sh -p scripts/train_baseline.py --task Digit-BaselineLSTM-v0 --headless

    # Optional: Resume from checkpoint
    ./isaaclab.sh -p scripts/train_baseline.py --task Digit-Baseline-v0 --resume \
        --checkpoint logs/digit_baseline/run_001/model_5000.pt --headless

Distributed training hyperparameter scaling:
    When --distributed is used, the following hyperparameters are automatically
    scaled to ensure faster convergence compared to single-GPU training:

    1. Learning rate: scaled by sqrt(world_size / 2).
       Rationale: gradients are averaged across GPUs via all-reduce, so each
       update uses a lower-variance gradient estimate. Full sqrt(N) scaling
       was found to be too aggressive for PPO with adaptive KL — it caused
       noise std to grow unchecked and reward to collapse after ~250 iters.
       sqrt(N/2) provides a safer balance between speed and stability.

    2. Mini-batches: scaled so each per-GPU mini-batch contains ~98K transitions,
       matching the single-GPU default (16384 envs * 48 steps / 8 mini-batches).
       This prevents excessively large mini-batches that waste the benefit of
       stochastic mini-batch updates within each PPO epoch.

    3. Learning epochs: increased by 1 (capped at 8). Too many epochs (e.g. +3)
       caused overfitting to recent experience and policy oscillation. A modest
       increase captures some benefit of diverse multi-GPU experience without
       destabilizing training.

    Example with 8x RTX 5090 and 262,144 total envs:
        Per GPU: 32,768 envs, 1.57M transitions/iter
        LR: 1e-3 -> 2.0e-3 (sqrt(8/2) scaling)
        Mini-batches: 8 -> 16 (1.57M / 98K)
        Learning epochs: 5 -> 6

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
parser.add_argument("--distributed", action="store_true", default=False,
                    help="Run training with multiple GPUs or nodes.")
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
    # Determine distributed training settings
    is_distributed = args_cli.distributed
    local_rank = 0
    global_rank = 0
    world_size = 1

    if is_distributed:
        local_rank = app_launcher.local_rank
        global_rank = app_launcher.global_rank
        world_size = int(os.getenv("WORLD_SIZE", 1))

    is_main_rank = (global_rank == 0)

    if is_main_rank:
        print(f"\n{'='*60}")
        print(f"BASELINE TRAINING (Radosavovic et al. 2024)")
        print(f"{'='*60}")
        print(f"Task: {args_cli.task}")
        print(f"Resume: {args_cli.resume}")
        if is_distributed:
            print(f"Distributed: {world_size} GPUs")
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

    # Configure device and env count for distributed training
    if is_distributed:
        env_cfg_obj.sim.device = f"cuda:{local_rank}"
        agent_cfg_obj.seed = agent_cfg_obj.seed + local_rank
        # Split environments across GPUs
        total_envs = env_cfg_obj.scene.num_envs
        env_cfg_obj.scene.num_envs = total_envs // world_size
        if is_main_rank:
            print(f"Distributed: {total_envs} total envs -> {env_cfg_obj.scene.num_envs} per GPU")

        # Scale hyperparameters for multi-GPU convergence
        import math
        base_lr = agent_cfg_obj.algorithm.learning_rate
        base_mini_batches = agent_cfg_obj.algorithm.num_mini_batches
        base_epochs = agent_cfg_obj.algorithm.num_learning_epochs

        # 1. Scale learning rate conservatively — sqrt(world_size) was too aggressive
        #    and caused reward collapse after ~250 iters (noise std grew unchecked).
        #    sqrt(world_size/2) provides a safer scaling that still benefits from
        #    the lower-variance gradients of multi-GPU all-reduce.
        agent_cfg_obj.algorithm.learning_rate = base_lr * math.sqrt(world_size / 2)

        # 2. Scale mini-batches to keep per-GPU mini-batch size similar to single-GPU
        #    default: 16384 envs * 48 steps / 8 mini-batches = ~98K per mini-batch
        envs_per_gpu = env_cfg_obj.scene.num_envs
        transitions_per_gpu = envs_per_gpu * agent_cfg_obj.num_steps_per_env
        target_minibatch_size = 98_304  # single-GPU default
        agent_cfg_obj.algorithm.num_mini_batches = max(
            4, round(transitions_per_gpu / target_minibatch_size)
        )

        # 3. Slight increase in learning epochs — too many (8) caused overfitting
        #    and policy oscillation. +1 is conservative but safe.
        agent_cfg_obj.algorithm.num_learning_epochs = min(base_epochs + 1, 8)

        if is_main_rank:
            print(f"  LR scaled: {base_lr:.1e} -> {agent_cfg_obj.algorithm.learning_rate:.2e}")
            print(f"  Mini-batches scaled: {base_mini_batches} -> {agent_cfg_obj.algorithm.num_mini_batches}")
            print(f"  Learning epochs scaled: {base_epochs} -> {agent_cfg_obj.algorithm.num_learning_epochs}")

    device = f"cuda:{local_rank}" if is_distributed else "cuda:0"

    # Setup logging directory (only rank 0 creates dirs and log files)
    log_root = os.path.join("logs", agent_cfg_obj.experiment_name)
    tee_logger = None

    if args_cli.resume and args_cli.checkpoint:
        log_dir = os.path.dirname(args_cli.checkpoint)
    else:
        # All ranks compute the same log_dir (same filesystem view before any mkdir)
        log_dir = get_next_run_dir(log_root)

    os.makedirs(log_dir, exist_ok=True)

    if is_main_rank:
        # Setup logging to file (rank 0 only)
        timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"training_{timestamp}.log")
        tee_logger = TeeLogger(log_file)
        sys.stdout = tee_logger

    # Set resume flag
    agent_cfg_obj.resume = args_cli.resume

    # Create environment
    if is_main_rank:
        print(f"Creating environment with {env_cfg_obj.scene.num_envs} envs (on {device})...")
    env = gym.make(args_cli.task, cfg=env_cfg_obj, render_mode=None)

    # Wrap environment for RSL-RL
    env = RslRlVecEnvWrapper(env)

    # Create runner
    if is_main_rank:
        print(f"Creating PPO runner...")
        print(f"  Policy type: {agent_cfg_obj.policy.__class__.__name__}")
        print(f"  Log directory: {log_dir}")
        print(f"  Device: {device}")

    runner = OnPolicyRunner(env, agent_cfg_obj.to_dict(), log_dir=log_dir, device=device)

    # Load checkpoint if resuming
    if args_cli.resume:
        checkpoint_path = args_cli.checkpoint
        if checkpoint_path is None:
            checkpoint_path = find_latest_checkpoint(log_dir)

        if checkpoint_path and os.path.exists(checkpoint_path):
            if is_main_rank:
                print(f"Loading checkpoint: {checkpoint_path}")
            runner.load(checkpoint_path)
        elif is_main_rank:
            print(f"Warning: No checkpoint found, starting from scratch")

    # Print training info
    if is_main_rank:
        print(f"\n{'='*60}")
        print(f"TRAINING CONFIGURATION")
        print(f"{'='*60}")
        print(f"Task: {args_cli.task}")
        print(f"Num envs per GPU: {env_cfg_obj.scene.num_envs}")
        if is_distributed:
            print(f"Total envs: {env_cfg_obj.scene.num_envs * world_size}")
            print(f"World size: {world_size}")
        print(f"Max iterations: {agent_cfg_obj.max_iterations}")
        print(f"Steps per env: {agent_cfg_obj.num_steps_per_env}")
        print(f"Log directory: {log_dir}")
        print(f"{'='*60}\n")

    # Start training
    if is_main_rank:
        print("Starting training...")
    start_time = dt.now()

    runner.learn(num_learning_iterations=agent_cfg_obj.max_iterations, init_at_random_ep_len=True)

    # Save final model
    final_model_path = os.path.join(log_dir, "model_final.pt")
    runner.save(final_model_path)

    end_time = dt.now()
    duration = end_time - start_time

    if is_main_rank:
        print(f"\n{'='*60}")
        print(f"TRAINING COMPLETE")
        print(f"{'='*60}")
        print(f"Duration: {duration}")
        print(f"Final model: {final_model_path}")
        print(f"{'='*60}\n")

    # Cleanup
    env.close()

    if tee_logger is not None:
        sys.stdout = tee_logger.terminal
        tee_logger.close()
        print(f"Log saved to: {log_file}")


if __name__ == "__main__":
    main()
    simulation_app.close()
