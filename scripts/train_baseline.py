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
import json
import os
import sys
import datetime
import re
import time
from collections import deque


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
parser.add_argument("--seed", type=int, default=None, help="Random seed (default: None = no seed fixing, use 42 for reproducibility)")
parser.add_argument("--max_iterations", type=int, default=None, help="Max training iterations")
parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path to resume from")
parser.add_argument("--agent_cfg", type=str, default=None,
                    help="Override agent config entry point (e.g. digit_locomotion.agents.baseline_cfg:DigitBaselineTeacherCommunityPPORunnerCfg)")
parser.add_argument("--load_config", type=str, default=None,
                    help="Load training config from a saved JSON file (reproduces exact training setup)")
parser.add_argument("--episode_length_s", type=float, default=None, help="Episode length in seconds")
parser.add_argument("--save_interval", type=int, default=10, help="Save checkpoint every N iterations (default: 10)")
parser.add_argument("--video", action="store_true", help="Record video during training")
parser.add_argument("--video_length", type=int, default=200, help="Video length in steps")
parser.add_argument("--video_interval", type=int, default=2000, help="Video recording interval")

# Adaptive LR arguments
parser.add_argument("--adaptive_lr", action="store_true", help="Enable adaptive learning rate with collapse detection")
parser.add_argument("--lr_min", type=float, default=1e-5, help="Minimum learning rate")
parser.add_argument("--lr_max", type=float, default=3e-3, help="Maximum learning rate")
parser.add_argument("--lr_decay_factor", type=float, default=0.5, help="LR decay factor on collapse")
parser.add_argument("--collapse_threshold", type=float, default=0.7, help="Episode length ratio to trigger LR reduction")
parser.add_argument("--rollback_threshold", type=float, default=0.4, help="Episode length ratio to trigger checkpoint rollback")

# Learning rate override
parser.add_argument("--learning_rate", type=float, default=None,
                    help="Override initial learning rate (default: from config)")
parser.add_argument("--lr_cap", type=float, default=5e-3,
                    help="Maximum learning rate cap for adaptive scheduler (default: 5e-3)")

# Custom reward weight arguments (None = use config default, 0.0 = disabled)
parser.add_argument("--upright_posture_weight", type=float, default=None,
                    help="Weight for upright posture reward (default: from config, use 0 to disable)")
parser.add_argument("--forward_lean_weight", type=float, default=None,
                    help="Weight for excessive forward lean penalty (default: from config, use 0 to disable)")
parser.add_argument("--arm_coordination_weight", type=float, default=None,
                    help="Weight for arm-leg coordination reward (default: from config, use 0 to disable)")
parser.add_argument("--arm_swing_bias_weight", type=float, default=None,
                    help="Weight for arm swing bias penalty (default: from config, use 0 to disable)")

# Append AppLauncher CLI args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# After app is launched, import remaining modules
import gymnasium as gym
import numpy as np
import random
import torch
from datetime import datetime as dt


def set_seed(seed: int):
    """Set random seeds for reproducibility across all random sources."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # NOTE: Do NOT set cudnn.deterministic=True or benchmark=False
        # — these caused steeper ep_length decline in earlier runs
    print(f"[Seed] Random seeds set to {seed} for reproducibility")

def save_training_config(log_dir: str, args_cli, env_cfg_obj, agent_cfg_obj):
    """Save complete training config to JSON for reproducibility."""
    agent_dict = agent_cfg_obj.to_dict()
    # @configclass to_dict() may not reflect instance-level overrides from CLI
    agent_dict["save_interval"] = agent_cfg_obj.save_interval
    agent_dict["max_iterations"] = agent_cfg_obj.max_iterations
    agent_dict["seed"] = agent_cfg_obj.seed

    config = {
        "timestamp": dt.now().isoformat(),
        "cli_args": vars(args_cli),
        "agent_config": agent_dict,
        "env_config": {
            "num_envs": env_cfg_obj.scene.num_envs,
            "episode_length_s": env_cfg_obj.episode_length_s,
            "decimation": env_cfg_obj.decimation,
        },
        "rewards": {},
    }

    # Serialize reward weights
    for name in dir(env_cfg_obj.rewards):
        attr = getattr(env_cfg_obj.rewards, name, None)
        if hasattr(attr, "weight"):
            reward_entry = {"weight": attr.weight}
            if hasattr(attr, "params") and attr.params:
                # Only serialize simple params (skip SceneEntityCfg objects)
                simple_params = {}
                for k, v in attr.params.items():
                    if isinstance(v, (int, float, str, bool)):
                        simple_params[k] = v
                if simple_params:
                    reward_entry["params"] = simple_params
            config["rewards"][name] = reward_entry

    # Clean up non-serializable CLI args
    cli_clean = {}
    for k, v in config["cli_args"].items():
        if isinstance(v, (int, float, str, bool, type(None), list)):
            cli_clean[k] = v
    config["cli_args"] = cli_clean

    config_path = os.path.join(log_dir, "training_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, default=str)
    print(f"[Config] Saved training config to: {config_path}")
    return config_path


from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

# Import Digit environments (triggers registration)
import digit_locomotion.tasks.locomotion  # noqa: F401

# Import custom transformer policy for RSL-RL integration
from digit_locomotion.networks.transformer_policy import ActorCriticTransformer  # noqa: F401


class MultiMetricMonitor:
    """Monitor multiple metrics for early warning of training collapse."""

    def __init__(self, env, window_size: int = 20, degradation_threshold: float = 1.3):
        self.env = env
        self.window_size = window_size
        self.degradation_threshold = degradation_threshold

        self.flat_orientation_history = []
        self.arm_roll_history = []
        self.ang_vel_xy_history = []
        self.is_alive_history = []

        self.best_flat_orientation = float('inf')
        self.best_arm_roll = float('inf')
        self.best_ang_vel_xy = float('inf')
        self.best_is_alive = 0.0

        self.warning_count = 0
        self.last_warning_iteration = 0

    def compute_metrics(self) -> dict:
        try:
            robot = self.env.unwrapped.scene["robot"]
            projected_gravity = robot.data.projected_gravity_b
            flat_orientation = torch.sum(torch.square(projected_gravity[:, :2]), dim=1).mean().item()
            ang_vel_xy = torch.sum(torch.square(robot.data.root_ang_vel_b[:, :2]), dim=1).mean().item()

            joint_names = robot.data.joint_names
            joint_pos = robot.data.joint_pos
            arm_roll = 0.0
            arm_roll_count = 0
            for i, name in enumerate(joint_names):
                if "shoulder" in name.lower() and "roll" in name.lower():
                    arm_roll += torch.abs(joint_pos[:, i]).mean().item()
                    arm_roll_count += 1
            if arm_roll_count > 0:
                arm_roll /= arm_roll_count

            episode_length_buf = self.env.unwrapped.episode_length_buf
            max_episode_length = self.env.unwrapped.max_episode_length
            is_alive = (episode_length_buf.float() / max_episode_length).mean().item()

            return {"flat_orientation": flat_orientation, "arm_roll": arm_roll,
                    "ang_vel_xy": ang_vel_xy, "is_alive": is_alive}
        except Exception:
            return {"flat_orientation": 0.0, "arm_roll": 0.0, "ang_vel_xy": 0.0, "is_alive": 1.0}

    def update(self, iteration: int) -> dict:
        metrics = self.compute_metrics()
        self.flat_orientation_history.append(metrics["flat_orientation"])
        self.arm_roll_history.append(metrics["arm_roll"])
        self.ang_vel_xy_history.append(metrics["ang_vel_xy"])
        self.is_alive_history.append(metrics["is_alive"])

        if metrics["flat_orientation"] < self.best_flat_orientation:
            self.best_flat_orientation = metrics["flat_orientation"]
        if metrics["arm_roll"] < self.best_arm_roll:
            self.best_arm_roll = metrics["arm_roll"]
        if metrics["ang_vel_xy"] < self.best_ang_vel_xy:
            self.best_ang_vel_xy = metrics["ang_vel_xy"]
        if metrics["is_alive"] > self.best_is_alive:
            self.best_is_alive = metrics["is_alive"]

        warning = None
        if len(self.flat_orientation_history) >= self.window_size:
            degraded_metrics = []
            current_orientation = sum(self.flat_orientation_history[-self.window_size:]) / self.window_size
            if self.best_flat_orientation > 0 and current_orientation > self.best_flat_orientation * self.degradation_threshold:
                degraded_metrics.append(f"flat_orientation ({current_orientation:.4f} vs best {self.best_flat_orientation:.4f})")

            current_arm_roll = sum(self.arm_roll_history[-self.window_size:]) / self.window_size
            if self.best_arm_roll > 0 and current_arm_roll > self.best_arm_roll * self.degradation_threshold:
                degraded_metrics.append(f"arm_roll ({current_arm_roll:.4f} vs best {self.best_arm_roll:.4f})")

            current_ang_vel = sum(self.ang_vel_xy_history[-self.window_size:]) / self.window_size
            if self.best_ang_vel_xy > 0 and current_ang_vel > self.best_ang_vel_xy * self.degradation_threshold:
                degraded_metrics.append(f"ang_vel_xy ({current_ang_vel:.4f} vs best {self.best_ang_vel_xy:.4f})")

            current_alive = sum(self.is_alive_history[-self.window_size:]) / self.window_size
            alive_threshold = self.best_is_alive * (2 - self.degradation_threshold)
            if current_alive < alive_threshold:
                degraded_metrics.append(f"is_alive ({current_alive:.4f} vs best {self.best_is_alive:.4f})")

            if len(degraded_metrics) >= 2 and iteration - self.last_warning_iteration > 50:
                self.warning_count += 1
                self.last_warning_iteration = iteration
                warning = {"level": "WARNING" if len(degraded_metrics) == 2 else "CRITICAL",
                           "degraded_metrics": degraded_metrics, "count": len(degraded_metrics)}

        metrics["warning"] = warning
        return metrics

    def get_status_string(self, metrics: dict) -> str:
        return (f"orient={metrics['flat_orientation']:.4f} arm_roll={metrics['arm_roll']:.4f} "
                f"ang_vel={metrics['ang_vel_xy']:.4f} alive={metrics['is_alive']:.3f}")


def train_with_adaptive_lr(runner, max_iterations: int, log_dir: str, args_cli):
    """Training with adaptive learning rate and collapse detection."""
    lr_min = args_cli.lr_min
    lr_max = args_cli.lr_max
    lr_decay_factor = args_cli.lr_decay_factor
    collapse_threshold = args_cli.collapse_threshold
    rollback_threshold = args_cli.rollback_threshold

    chunk_size = 50
    num_chunks = (max_iterations + chunk_size - 1) // chunk_size
    current_lr = runner.alg.optimizer.param_groups[0]['lr']

    ep_length_history = deque(maxlen=200)
    peak_ep_length = 0.0
    best_ep_length = 0.0
    best_checkpoint_path = None
    lr_reductions = 0
    rollbacks = 0
    total_iterations = 0
    metric_warnings = 0

    metric_monitor = MultiMetricMonitor(env=runner.env, window_size=20, degradation_threshold=1.3)

    print(f"\n{'='*60}")
    print("ADAPTIVE LEARNING RATE ENABLED")
    print(f"{'='*60}")
    print(f"  LR range: [{lr_min:.2e}, {lr_max:.2e}]")
    print(f"  Decay factor: {lr_decay_factor}x on collapse")
    print(f"  Collapse threshold: {collapse_threshold*100:.0f}% of peak")
    print(f"  Rollback threshold: {rollback_threshold*100:.0f}% of peak")
    print(f"  Check interval: {chunk_size} iterations")
    print(f"{'='*60}\n")

    start_time = time.time()

    for chunk_idx in range(num_chunks):
        target_iteration = min((chunk_idx + 1) * chunk_size, max_iterations)
        if target_iteration <= total_iterations:
            break

        iters_this_chunk = target_iteration - total_iterations
        print(f"\n--- Chunk {chunk_idx + 1}/{num_chunks}: iterations {total_iterations + 1}-{target_iteration} ({iters_this_chunk} iters) ---")
        runner.learn(num_learning_iterations=iters_this_chunk, init_at_random_ep_len=(chunk_idx == 0))
        # RSL-RL bug: learn() sets current_learning_iteration = it (loop var),
        # not it+1. Increment so next chunk starts at the right iteration.
        runner.current_learning_iteration += 1
        total_iterations = target_iteration

        try:
            ep_len = runner.env.episode_length_buf.float().mean().item()
        except:
            ep_len = 100.0

        ep_length_history.append(ep_len)
        if ep_len > peak_ep_length:
            peak_ep_length = ep_len
        if ep_len > best_ep_length:
            best_ep_length = ep_len
            best_checkpoint_path = os.path.join(log_dir, f"model_best.pt")
            runner.save(best_checkpoint_path)

        if len(ep_length_history) >= 10:
            current_avg = sum(list(ep_length_history)[-10:]) / 10
        else:
            current_avg = ep_len

        ratio_to_peak = current_avg / max(1.0, peak_ep_length)

        metrics = metric_monitor.update(total_iterations)
        metric_status = metric_monitor.get_status_string(metrics)

        early_warning_triggered = False
        if metrics["warning"]:
            warning = metrics["warning"]
            metric_warnings += 1
            print(f"\n{'*'*60}")
            print(f"[Multi-Metric] {warning['level']}: {warning['count']} metrics degraded!")
            for m in warning["degraded_metrics"]:
                print(f"  - {m}")
            print(f"{'*'*60}")

            if warning["level"] == "CRITICAL":
                early_warning_triggered = True
                new_lr = max(lr_min, current_lr * lr_decay_factor)
                for param_group in runner.alg.optimizer.param_groups:
                    param_group['lr'] = new_lr
                runner.alg.learning_rate = new_lr  # Sync adaptive scheduler
                print(f"[Multi-Metric] Preemptive LR reduction: {current_lr:.2e} -> {new_lr:.2e}")
                current_lr = new_lr
                lr_reductions += 1

        if not early_warning_triggered and ratio_to_peak < rollback_threshold and best_checkpoint_path and len(ep_length_history) > 20:
            print(f"\n{'!'*60}")
            print(f"[Adaptive LR] SEVERE COLLAPSE DETECTED!")
            print(f"  Current avg: {current_avg:.1f}, Peak: {peak_ep_length:.1f} (ratio: {ratio_to_peak:.2f})")
            print(f"  Rolling back to best checkpoint")
            print(f"{'!'*60}\n")
            runner.load(best_checkpoint_path)
            new_lr = max(lr_min, current_lr * lr_decay_factor)
            for param_group in runner.alg.optimizer.param_groups:
                param_group['lr'] = new_lr
            runner.alg.learning_rate = new_lr  # Sync adaptive scheduler
            current_lr = new_lr
            lr_reductions += 1
            rollbacks += 1
            ep_length_history.clear()
            peak_ep_length = best_ep_length

        elif not early_warning_triggered and ratio_to_peak < collapse_threshold and len(ep_length_history) > 10:
            new_lr = max(lr_min, current_lr * lr_decay_factor)
            for param_group in runner.alg.optimizer.param_groups:
                param_group['lr'] = new_lr
            runner.alg.learning_rate = new_lr  # Sync adaptive scheduler
            print(f"[Adaptive LR] Collapse detected, reducing LR: {current_lr:.2e} -> {new_lr:.2e}")
            current_lr = new_lr
            lr_reductions += 1

        print(f"[Adaptive LR] Status: LR={current_lr:.2e}, EP len avg={current_avg:.1f}, Peak={peak_ep_length:.1f}")
        print(f"[Multi-Metric] {metric_status}")

    elapsed = time.time() - start_time

    print(f"\n{'='*60}")
    print("ADAPTIVE LR TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Total iterations: {total_iterations}")
    print(f"  Total time: {elapsed/60:.1f} minutes")
    print(f"  Final LR: {current_lr:.2e}")
    print(f"  Peak episode length: {peak_ep_length:.0f}")
    print(f"  Best episode length: {best_ep_length:.0f}")
    print(f"  LR reductions: {lr_reductions}")
    print(f"  Checkpoint rollbacks: {rollbacks}")
    print(f"  Metric warnings: {metric_warnings}")
    print(f"{'='*60}\n")

    return best_checkpoint_path


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
    agent_cfg = args_cli.agent_cfg if args_cli.agent_cfg else gym.spec(args_cli.task).kwargs["rsl_rl_cfg_entry_point"]
    if args_cli.agent_cfg:
        print(f"[Config Override] Using agent config: {args_cli.agent_cfg}")

    # Parse configs
    env_cfg_class = env_cfg.rsplit(":", 1)
    env_cfg_module = __import__(env_cfg_class[0], fromlist=[env_cfg_class[1]])
    env_cfg_obj = getattr(env_cfg_module, env_cfg_class[1])()

    agent_cfg_class = agent_cfg.rsplit(":", 1)
    agent_cfg_module = __import__(agent_cfg_class[0], fromlist=[agent_cfg_class[1]])
    agent_cfg_obj = getattr(agent_cfg_module, agent_cfg_class[1])()

    # Load saved config to reproduce a previous run
    if args_cli.load_config:
        print(f"[Config] Loading config from: {args_cli.load_config}")
        with open(args_cli.load_config, "r") as f:
            saved_config = json.load(f)

        # Apply agent config overrides from saved file
        saved_agent = saved_config.get("agent_config", {})
        if "seed" in saved_agent:
            agent_cfg_obj.seed = saved_agent["seed"]
        if "max_iterations" in saved_agent:
            agent_cfg_obj.max_iterations = saved_agent["max_iterations"]
        if "num_steps_per_env" in saved_agent:
            agent_cfg_obj.num_steps_per_env = saved_agent["num_steps_per_env"]

        saved_policy = saved_agent.get("policy", {})
        if "init_noise_std" in saved_policy:
            agent_cfg_obj.policy.init_noise_std = saved_policy["init_noise_std"]

        saved_algo = saved_agent.get("algorithm", {})
        algo_fields = ["learning_rate", "gamma", "lam", "entropy_coef", "clip_param",
                        "num_learning_epochs", "num_mini_batches", "max_grad_norm",
                        "desired_kl", "value_loss_coef"]
        for field in algo_fields:
            if field in saved_algo:
                setattr(agent_cfg_obj.algorithm, field, saved_algo[field])

        # Apply env config overrides
        saved_env = saved_config.get("env_config", {})
        if "num_envs" in saved_env:
            env_cfg_obj.scene.num_envs = saved_env["num_envs"]
        if "episode_length_s" in saved_env:
            env_cfg_obj.episode_length_s = saved_env["episode_length_s"]

        # Apply reward weight overrides
        saved_rewards = saved_config.get("rewards", {})
        for reward_name, reward_info in saved_rewards.items():
            if hasattr(env_cfg_obj.rewards, reward_name):
                reward_term = getattr(env_cfg_obj.rewards, reward_name)
                reward_term.weight = reward_info["weight"]

        print(f"[Config] Loaded config from: {args_cli.load_config}")

    # Override with CLI args (CLI args take precedence over loaded config)
    if args_cli.num_envs is not None:
        env_cfg_obj.scene.num_envs = args_cli.num_envs

    # Set seed for reproducibility (None = no seed fixing, matches run_005 behavior)
    seed = args_cli.seed
    if seed is not None:
        agent_cfg_obj.seed = seed
        set_seed(seed)
    else:
        print("[Seed] No seed set - using non-deterministic training (matches run_005)")
    if args_cli.max_iterations is not None:
        agent_cfg_obj.max_iterations = args_cli.max_iterations
    agent_cfg_obj.save_interval = args_cli.save_interval
    if args_cli.episode_length_s is not None:
        env_cfg_obj.episode_length_s = args_cli.episode_length_s
    if args_cli.learning_rate is not None:
        agent_cfg_obj.algorithm.learning_rate = args_cli.learning_rate
        print(f"[LR Override] Initial learning rate set to {args_cli.learning_rate}")

    # Override custom reward weights if specified (only if terms exist in config)
    reward_overrides = []
    for arg_name, attr_name in [
        ("upright_posture_weight", "upright_posture"),
        ("forward_lean_weight", "excessive_forward_lean"),
        ("arm_coordination_weight", "arm_leg_coordination"),
        ("arm_swing_bias_weight", "arm_swing_bias"),
    ]:
        weight = getattr(args_cli, arg_name, None)
        if weight is not None:
            if hasattr(env_cfg_obj.rewards, attr_name):
                getattr(env_cfg_obj.rewards, attr_name).weight = weight
                reward_overrides.append(f"{attr_name}={weight}")
            else:
                print(f"[Warning] Reward term '{attr_name}' not in config, skipping override")

    if reward_overrides:
        print(f"Reward weight overrides: {', '.join(reward_overrides)}")

    # Setup logging directory — always create a new run directory
    log_root = os.path.join("logs", agent_cfg_obj.experiment_name)
    log_dir = get_next_run_dir(log_root)

    os.makedirs(log_dir, exist_ok=True)

    # Setup logging to file
    timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")
    tee_logger = TeeLogger(log_file)
    sys.stdout = tee_logger

    # Set resume flag
    agent_cfg_obj.resume = args_cli.resume

    # Save training config for reproducibility
    save_training_config(log_dir, args_cli, env_cfg_obj, agent_cfg_obj)

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
            # Override optimizer LR if explicitly set via CLI
            # (runner.load restores saved optimizer state, overwriting config LR)
            if args_cli.learning_rate is not None:
                for param_group in runner.alg.optimizer.param_groups:
                    param_group['lr'] = args_cli.learning_rate
                runner.alg.learning_rate = args_cli.learning_rate  # Sync adaptive scheduler's internal LR
                print(f"[LR Override] Optimizer + scheduler LR reset to {args_cli.learning_rate} after checkpoint load")
        else:
            print(f"Warning: No checkpoint found, starting from scratch")

    # Print training info
    print(f"\n{'='*60}")
    print(f"TRAINING CONFIGURATION")
    print(f"{'='*60}")
    print(f"Task: {args_cli.task}")
    print(f"Seed: {seed}")
    print(f"Num envs: {env_cfg_obj.scene.num_envs}")
    print(f"Max iterations: {agent_cfg_obj.max_iterations}")
    print(f"Episode length: {env_cfg_obj.episode_length_s}s")
    print(f"Steps per env: {agent_cfg_obj.num_steps_per_env}")
    print(f"Initial LR: {agent_cfg_obj.algorithm.learning_rate}")
    print(f"LR Cap: {args_cli.lr_cap}")
    print(f"Log directory: {log_dir}")
    print(f"{'='*60}\n")

    # Start training
    print("Starting training...")
    start_time = dt.now()

    if args_cli.adaptive_lr:
        # Use adaptive learning rate training with collapse detection
        best_checkpoint = train_with_adaptive_lr(runner, agent_cfg_obj.max_iterations, log_dir, args_cli)
        final_model_path = best_checkpoint if best_checkpoint else os.path.join(log_dir, "model_final.pt")
    else:
        # Standard training with LR capping every iteration.
        # CRITICAL: runner.learn(num_learning_iterations=N) is RELATIVE — it runs N
        # iterations from the current position. We MUST pass 1 each time so our LR cap
        # runs between every single training step. Passing iter_idx+1 (the old code)
        # caused the scheduler to spike LR during multi-iteration learn() calls.
        lr_cap = args_cli.lr_cap
        max_iterations = agent_cfg_obj.max_iterations
        lr_capped_count = 0

        print(f"[LR Cap] Maximum learning rate: {lr_cap}")

        # Disable RSL-RL's adaptive LR scheduler. It adjusts LR 40 times per
        # iteration (8 mini-batches × 5 epochs) inside update(), overriding our
        # external cap. Setting desired_kl=None makes the scheduler skip entirely.
        runner.alg.desired_kl = None
        runner.alg.schedule = "fixed"
        print(f"[LR Cap] Disabled adaptive LR scheduler (desired_kl=None, schedule=fixed)")

        start_global_iter = runner.current_learning_iteration
        total_target_iter = start_global_iter + max_iterations

        # Linear warmup from initial LR to lr_cap over first 50 iterations
        warmup_iters = 50
        start_lr = runner.alg.optimizer.param_groups[0]['lr']
        print(f"[LR Warmup] {start_lr:.2e} -> {lr_cap:.2e} over {warmup_iters} iterations")

        for iter_idx in range(max_iterations):
            # Apply LR warmup
            if iter_idx < warmup_iters:
                warmup_lr = start_lr + (lr_cap - start_lr) * (iter_idx / warmup_iters)
                for param_group in runner.alg.optimizer.param_groups:
                    param_group['lr'] = warmup_lr
                runner.alg.learning_rate = warmup_lr

            print(f"\n{'='*80}")
            current_lr = runner.alg.optimizer.param_groups[0]['lr']
            print(f"  Training Progress: {iter_idx + 1}/{max_iterations}  |  Global iteration: {runner.current_learning_iteration}/{total_target_iter}  |  LR: {current_lr:.2e}")
            print(f"{'='*80}")
            runner.learn(num_learning_iterations=1, init_at_random_ep_len=(iter_idx == 0))
            # RSL-RL bug: learn() sets current_learning_iteration = it (loop var),
            # not it+1. So after learn(1), the counter stays at the same value.
            # We must increment it ourselves so checkpoints save with correct names
            # and the iteration display progresses.
            runner.current_learning_iteration += 1

            # Cap learning rate every iteration
            # IMPORTANT: Must update BOTH the optimizer param_groups AND the
            # algorithm's self.learning_rate. RSL-RL's adaptive scheduler uses
            # self.learning_rate internally and overwrites optimizer LR each step.
            # If we only cap the optimizer, the scheduler ignores our cap.
            current_lr = runner.alg.optimizer.param_groups[0]['lr']
            if current_lr > lr_cap:
                for param_group in runner.alg.optimizer.param_groups:
                    param_group['lr'] = lr_cap
                runner.alg.learning_rate = lr_cap  # Sync the scheduler's internal LR
                lr_capped_count += 1
                if lr_capped_count <= 10 or lr_capped_count % 100 == 0:
                    print(f"[LR Cap] Iteration {iter_idx + 1}: LR {current_lr:.2e} -> {lr_cap:.2e} (capped)")

        if lr_capped_count > 0:
            print(f"[LR Cap] Total times capped: {lr_capped_count}/{max_iterations}")

        final_model_path = os.path.join(log_dir, "model_final.pt")

    # Save final model
    runner.save(os.path.join(log_dir, "model_final.pt"))

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
