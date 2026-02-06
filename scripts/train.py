#!/usr/bin/env python3
"""Training script for Digit locomotion policy.

This script trains a locomotion policy for the Agility Robotics Digit
robot using Isaac Lab and RSL-RL's PPO implementation.

Usage (Linux/macOS):
    # Train on flat terrain (recommended for initial training)
    ./isaaclab.sh -p scripts/train.py --task Digit-Velocity-Flat-v0

    # Train with custom settings
    ./isaaclab.sh -p scripts/train.py --task Digit-Velocity-Flat-v0 \\
        --num_envs 4096 --max_iterations 20000 --headless

    # Resume from checkpoint
    ./isaaclab.sh -p scripts/train.py --task Digit-Velocity-Flat-v0 \\
        --resume --checkpoint logs/digit_flat/run_001/model_5000.pt

Usage (Windows PowerShell):
    # IMPORTANT: In PowerShell, use .\\ prefix and deactivate conda first
    cd C:\\IsaacLab
    $env:CONDA_PREFIX = ""

    # New training run (creates logs/digit_flat/run_001/)
    .\\isaaclab.bat -p scripts\\train.py --task Digit-Velocity-Flat-v0 --headless

    # Resume from checkpoint (continues in same run directory)
    .\\isaaclab.bat -p scripts\\train.py --task Digit-Velocity-Flat-v0 --headless `
        --resume --checkpoint C:\\IsaacLab\\logs\\digit_flat\\run_001\\model_5000.pt

    # Resume from latest checkpoint in a run
    .\\isaaclab.bat -p scripts\\train.py --task Digit-Velocity-Flat-v0 --headless `
        --resume --run_dir C:\\IsaacLab\\logs\\digit_flat\\run_001

Usage (Windows Command Prompt):
    cd C:\\IsaacLab
    set CONDA_PREFIX=
    isaaclab.bat -p scripts\\train.py --task Digit-Velocity-Flat-v0 --headless
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
        "--checkpoint",
        type=str,
        default=None,
        help="Path to specific checkpoint file to resume from",
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="Path to run directory to resume from (uses latest checkpoint)",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Custom name for this run (default: auto-generated run_XXX)",
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

    # Reward scheduling
    parser.add_argument(
        "--reward_schedule",
        type=str,
        default=None,
        choices=["jogging", "conservative", None],
        help="Use reward scheduling to gradually introduce arm posture rewards",
    )
    parser.add_argument(
        "--schedule_warmup",
        type=int,
        default=200,
        help="Iterations before starting to introduce scheduled rewards",
    )
    parser.add_argument(
        "--schedule_duration",
        type=int,
        default=300,
        help="Iterations over which to ramp up scheduled rewards",
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
from datetime import datetime

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg, load_cfg_from_registry

# Import to register environments
import digit_locomotion  # noqa: F401

# Import reward scheduler
from digit_locomotion.tasks.locomotion.reward_scheduler import (
    RewardScheduler,
    create_jogging_scheduler,
    create_conservative_scheduler,
)


class ScheduledOnPolicyRunner:
    """Wrapper around OnPolicyRunner that supports reward scheduling.

    This class wraps the standard RSL-RL OnPolicyRunner and adds support
    for dynamically adjusting reward weights during training based on a
    schedule. This allows locomotion to stabilize before introducing
    arm posture constraints.
    """

    def __init__(
        self,
        env,
        train_cfg: dict,
        log_dir: str,
        device: str,
        scheduler: RewardScheduler | None = None,
    ):
        from rsl_rl.runners import OnPolicyRunner

        self.base_runner = OnPolicyRunner(
            env=env,
            train_cfg=train_cfg,
            log_dir=log_dir,
            device=device,
        )
        self.scheduler = scheduler
        self.env = env  # Keep reference for scheduling

    def load(self, path: str):
        """Load checkpoint."""
        self.base_runner.load(path)

    def save(self, path: str):
        """Save checkpoint."""
        self.base_runner.save(path)

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = True):
        """Run training with reward scheduling.

        This method runs the training loop with periodic reward weight updates
        based on the configured schedule.
        """
        if self.scheduler is None:
            # No scheduling - use standard training
            self.base_runner.learn(
                num_learning_iterations=num_learning_iterations,
                init_at_random_ep_len=init_at_random_ep_len,
            )
            return

        # Training with reward scheduling
        # We'll run training in chunks and update rewards between chunks
        print("\n" + "=" * 60)
        print("REWARD SCHEDULING ENABLED")
        print("=" * 60)
        print("Scheduled rewards will be gradually introduced:")
        for name, schedule in self.scheduler.schedules.items():
            print(f"  {name}: {schedule.start_weight:.3f} -> {schedule.end_weight:.3f}")
            print(f"    (iterations {schedule.start_iteration} to {schedule.end_iteration})")
        print("=" * 60 + "\n")

        # Get the unwrapped environment for reward updates
        base_env = self.env.unwrapped

        # Apply initial schedule (iteration 0)
        self.scheduler.update(base_env, 0)

        # Use the base runner's learn method but with a hook
        # We'll patch the alg's update method to include scheduling
        original_update = self.base_runner.alg.update

        def update_with_scheduling():
            """Wrapper that applies reward scheduling after each update."""
            result = original_update()
            # Get current iteration from runner
            current_iter = self.base_runner.current_learning_iteration
            self.scheduler.update(base_env, current_iter)
            return result

        # Patch the update method
        self.base_runner.alg.update = update_with_scheduling

        try:
            # Run training
            self.base_runner.learn(
                num_learning_iterations=num_learning_iterations,
                init_at_random_ep_len=init_at_random_ep_len,
            )
        finally:
            # Restore original method
            self.base_runner.alg.update = original_update


class TeeLogger:
    """Duplicates stdout to both console and a log file."""

    def __init__(self, log_file: str):
        self.terminal = sys.stdout
        self.log_file = open(log_file, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()  # Ensure immediate write

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()


def get_next_run_dir(base_dir: str, experiment_name: str) -> str:
    """Get the next available run directory (run_001, run_002, etc.)."""
    experiment_dir = os.path.join(base_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)

    # Find existing run directories
    existing_runs = []
    if os.path.exists(experiment_dir):
        for name in os.listdir(experiment_dir):
            if name.startswith("run_") and os.path.isdir(os.path.join(experiment_dir, name)):
                try:
                    run_num = int(name.split("_")[1])
                    existing_runs.append(run_num)
                except (IndexError, ValueError):
                    pass

    # Get next run number
    next_run = max(existing_runs, default=0) + 1
    return os.path.join(experiment_dir, f"run_{next_run:03d}")


def find_latest_checkpoint(run_dir: str) -> str | None:
    """Find the latest checkpoint in a run directory."""
    if not os.path.exists(run_dir):
        return None

    checkpoints = []
    for name in os.listdir(run_dir):
        if name.startswith("model_") and name.endswith(".pt"):
            try:
                # Extract iteration number from model_XXXX.pt
                iter_num = int(name.replace("model_", "").replace(".pt", ""))
                checkpoints.append((iter_num, os.path.join(run_dir, name)))
            except ValueError:
                pass

    if not checkpoints:
        # Check for final_model.pt
        final_model = os.path.join(run_dir, "final_model.pt")
        if os.path.exists(final_model):
            return final_model
        return None

    # Return checkpoint with highest iteration number
    checkpoints.sort(key=lambda x: x[0], reverse=True)
    return checkpoints[0][1]


def main():
    """Main training function."""
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

    # Override configuration with command line arguments
    if args.max_iterations is not None:
        agent_cfg.max_iterations = args.max_iterations
    if args.experiment_name is not None:
        agent_cfg.experiment_name = args.experiment_name

    # Determine checkpoint to load (if resuming)
    checkpoint_path = None
    if args.resume:
        if args.checkpoint:
            checkpoint_path = args.checkpoint
        elif args.run_dir:
            checkpoint_path = find_latest_checkpoint(args.run_dir)
            if checkpoint_path is None:
                raise FileNotFoundError(f"No checkpoint found in {args.run_dir}")
        else:
            raise ValueError("--resume requires either --checkpoint or --run_dir")

        print(f"Resuming from checkpoint: {checkpoint_path}")

    # Determine log directory
    if args.resume and args.run_dir:
        # Continue in the same run directory
        log_dir = args.run_dir
    elif args.resume and args.checkpoint:
        # Continue in the same directory as the checkpoint
        log_dir = os.path.dirname(args.checkpoint)
    elif args.run_name:
        # Use custom run name
        log_dir = os.path.join(args.log_dir, agent_cfg.experiment_name, args.run_name)
    else:
        # Create new run directory with incrementing number
        log_dir = get_next_run_dir(args.log_dir, agent_cfg.experiment_name)

    os.makedirs(log_dir, exist_ok=True)

    # Setup logging to file (tee stdout to both console and log file)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")
    tee_logger = TeeLogger(log_file)
    sys.stdout = tee_logger
    print(f"Logging training output to: {log_file}")

    # Create reward scheduler if requested
    scheduler = None
    if args.reward_schedule:
        if args.reward_schedule == "jogging":
            scheduler = create_jogging_scheduler(
                locomotion_warmup=args.schedule_warmup,
                arm_transition_duration=args.schedule_duration,
            )
        elif args.reward_schedule == "conservative":
            scheduler = create_conservative_scheduler(
                locomotion_warmup=args.schedule_warmup,
                arm_transition_duration=args.schedule_duration,
            )
        print(f"Reward scheduling: {args.reward_schedule}")
        print(f"  Warmup iterations: {args.schedule_warmup}")
        print(f"  Transition duration: {args.schedule_duration}")

    # Create the runner (with or without scheduling)
    if scheduler is not None:
        runner = ScheduledOnPolicyRunner(
            env=env,
            train_cfg=agent_cfg.to_dict(),
            log_dir=log_dir,
            device=env.device,
            scheduler=scheduler,
        )
    else:
        # Import and use standard RSL-RL runner
        from rsl_rl.runners import OnPolicyRunner
        runner = OnPolicyRunner(
            env=env,
            train_cfg=agent_cfg.to_dict(),
            log_dir=log_dir,
            device=env.device,
        )

    # Load checkpoint if resuming
    if checkpoint_path:
        print(f"Loading checkpoint: {checkpoint_path}")
        runner.load(checkpoint_path)

    # Print training info
    print("\n" + "=" * 60)
    print("DIGIT LOCOMOTION TRAINING")
    print("=" * 60)
    print(f"Task: {args.task}")
    print(f"Number of environments: {env.num_envs}")
    print(f"Max iterations: {agent_cfg.max_iterations}")
    print(f"Log directory: {log_dir}")
    print(f"Device: {env.device}")
    if checkpoint_path:
        print(f"Resumed from: {checkpoint_path}")
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
    print(f"\nLog file saved to: {log_file}")
    sys.stdout = tee_logger.terminal  # Restore original stdout
    tee_logger.close()
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
