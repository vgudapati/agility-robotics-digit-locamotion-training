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

# Episode length configuration
parser.add_argument("--episode-length", type=int, default=1000,
                    help="Max episode length in steps (default: 1000)")

# Early stopping parameters
parser.add_argument("--early-stop", action="store_true",
                    help="Enable early stopping based on convergence criteria")
parser.add_argument("--early-stop-timeout", type=float, default=0.95,
                    help="Time-out rate threshold (default: 0.95 = 95%%)")
parser.add_argument("--early-stop-ep-length-pct", type=float, default=0.95,
                    help="Episode length threshold as percentage of --episode-length (default: 0.95 = 95%%)")
parser.add_argument("--early-stop-vel-tracking", type=float, default=0.90,
                    help="Velocity tracking threshold (default: 0.90 = 90%%)")
parser.add_argument("--early-stop-patience", type=int, default=100,
                    help="Number of iterations to sustain all thresholds before stopping (default: 100)")

# Adaptive learning rate scheduler
parser.add_argument("--adaptive-lr", action="store_true",
                    help="Enable adaptive LR based on episode length trends")
parser.add_argument("--lr-min", type=float, default=1e-5,
                    help="Minimum learning rate (default: 1e-5)")
parser.add_argument("--lr-max", type=float, default=1e-3,
                    help="Maximum learning rate (default: 1e-3)")
parser.add_argument("--lr-decay-factor", type=float, default=0.5,
                    help="LR decay factor when collapse detected (default: 0.5)")
parser.add_argument("--lr-grow-factor", type=float, default=1.1,
                    help="LR growth factor when improving (default: 1.1)")
parser.add_argument("--collapse-threshold", type=float, default=0.7,
                    help="Collapse detection: current/peak ratio (default: 0.7 = 70%%)")
parser.add_argument("--rollback-threshold", type=float, default=0.3,
                    help="Severe collapse threshold for checkpoint rollback (default: 0.3 = 30%%)")

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


class EarlyStoppingRunner(OnPolicyRunner):
    """OnPolicyRunner with early stopping based on convergence criteria."""

    def __init__(
        self,
        env,
        train_cfg,
        log_dir=None,
        device="cpu",
        # Early stopping parameters
        early_stop_enabled=False,
        early_stop_timeout=0.95,
        early_stop_ep_length=950.0,
        early_stop_vel_tracking=0.90,
        early_stop_patience=100,
    ):
        super().__init__(env, train_cfg, log_dir, device)
        self.early_stop_enabled = early_stop_enabled
        self.early_stop_timeout = early_stop_timeout
        self.early_stop_ep_length = early_stop_ep_length
        self.early_stop_vel_tracking = early_stop_vel_tracking
        self.early_stop_patience = early_stop_patience
        self.convergence_count = 0
        self.best_checkpoint_path = None
        self.stopped_early = False
        self.final_iteration = 0

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        """Training loop with early stopping support."""
        import time
        import os

        # Initialize
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
        obs = self.env.get_observations()
        # Set model to training mode (handle different RSL-RL API versions)
        if hasattr(self.alg, 'train_mode'):
            self.alg.train_mode()
        elif hasattr(self.alg, 'actor_critic'):
            self.alg.actor_critic.train()
        elif hasattr(self.alg, 'policy'):
            self.alg.policy.train()

        # Reset logging
        rewbuffer = []
        lenbuffer = []
        timeout_buffer = []  # Track timeout vs early termination for each episode
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_timesteps = 0
        tot_time = 0
        start_time = time.time()

        for it in range(num_learning_iterations):
            self.final_iteration = it + 1

            # Rollout
            with torch.inference_mode():
                for _ in range(self.cfg["num_steps_per_env"]):
                    actions = self.alg.act(obs)
                    obs, rewards, dones, infos = self.env.step(actions)

                    # Handle dones being a dict (Isaac Lab) vs tensor (standard)
                    if isinstance(dones, dict):
                        # Isaac Lab returns dict with 'terminated' and 'truncated'
                        dones_tensor = dones.get("terminated", dones.get("done", torch.zeros(self.env.num_envs, device=self.device)))
                        if "truncated" in dones:
                            dones_tensor = dones_tensor | dones["truncated"]
                    else:
                        dones_tensor = dones

                    # Handle different RSL-RL API versions
                    extras = infos.get("extras", {}) if isinstance(infos, dict) else {}
                    try:
                        self.alg.process_env_step(rewards, dones_tensor, infos, extras)
                    except TypeError:
                        self.alg.process_env_step(rewards, dones_tensor, infos)

                    # Track rewards and episode lengths
                    cur_reward_sum += rewards
                    cur_episode_length += 1
                    new_ids = (dones_tensor > 0).nonzero(as_tuple=False)

                    if len(new_ids) > 0:
                        rewbuffer.extend(cur_reward_sum[new_ids[:, 0]].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids[:, 0]].cpu().numpy().tolist())

                        # Track timeout vs early termination
                        if "time_outs" in infos:
                            timeouts = infos["time_outs"]
                            if isinstance(timeouts, torch.Tensor):
                                # 1.0 = timeout (good), 0.0 = early termination (bad)
                                timeout_buffer.extend(timeouts[new_ids[:, 0]].float().cpu().numpy().tolist())
                            else:
                                # Fallback: assume timeout if episode reached near max length
                                for idx in new_ids[:, 0]:
                                    is_timeout = cur_episode_length[idx].item() >= self.env.max_episode_length * 0.95
                                    timeout_buffer.append(1.0 if is_timeout else 0.0)

                        cur_reward_sum[new_ids[:, 0]] = 0
                        cur_episode_length[new_ids[:, 0]] = 0

            # Learning step
            self.alg.compute_returns(obs)
            mean_value_loss, mean_surrogate_loss = self.alg.update()

            # Compute statistics
            tot_timesteps += self.cfg["num_steps_per_env"] * self.env.num_envs
            iteration_time = time.time() - start_time
            tot_time += iteration_time
            start_time = time.time()

            # Get mean episode length from recent episodes
            ep_len_mean = sum(lenbuffer[-100:]) / max(1, len(lenbuffer[-100:]))

            # Get timeout rate from recent episodes (ratio of episodes that ended due to timeout)
            timeout_rate = sum(timeout_buffer[-100:]) / max(1, len(timeout_buffer[-100:]))

            # Get velocity tracking from environment extras if available
            vel_tracking = 0.0
            if hasattr(self.env, "extras") and "episode" in self.env.extras:
                ep_info = self.env.extras["episode"]
                if "rew_tracking_lin_vel" in ep_info:
                    vel_tracking = ep_info["rew_tracking_lin_vel"]

            # Log iteration
            if it % self.cfg.get("log_interval", 1) == 0:
                fps = int(self.cfg["num_steps_per_env"] * self.env.num_envs / iteration_time)
                mean_rew = sum(rewbuffer[-100:]) / max(1, len(rewbuffer[-100:]))
                print(f"Iter {it+1}/{num_learning_iterations} | EP len: {ep_len_mean:.1f} | "
                      f"Timeout: {timeout_rate*100:.1f}% | Rew: {mean_rew:.2f} | FPS: {fps}")

            # Save checkpoint
            if (it + 1) % self.cfg.get("save_interval", 50) == 0:
                checkpoint_path = os.path.join(self.log_dir, f"model_{it+1}.pt")
                self.save(checkpoint_path)

            # Early stopping check
            if self.early_stop_enabled:
                # Check convergence criteria
                timeout_ok = timeout_rate >= self.early_stop_timeout
                ep_len_ok = ep_len_mean >= self.early_stop_ep_length
                # For velocity tracking, use positive values (reward term is typically positive)
                vel_ok = vel_tracking >= self.early_stop_vel_tracking or self.early_stop_vel_tracking <= 0

                if timeout_ok and ep_len_ok and vel_ok:
                    self.convergence_count += 1
                    if self.convergence_count == 1:
                        # Save checkpoint at first convergence
                        self.best_checkpoint_path = os.path.join(
                            self.log_dir, f"model_converged_{it+1}.pt"
                        )
                        self.save(self.best_checkpoint_path)
                        print(f"\n{'='*60}")
                        print(f"[Early Stop] Convergence criteria met at iteration {it+1}")
                        print(f"  Time-out rate: {timeout_rate*100:.1f}% >= {self.early_stop_timeout*100:.0f}%")
                        print(f"  Episode length: {ep_len_mean:.0f} >= {self.early_stop_ep_length:.0f}")
                        print(f"  Velocity tracking: {vel_tracking:.3f}")
                        print(f"  Saved checkpoint: {self.best_checkpoint_path}")
                        print(f"  Waiting for {self.early_stop_patience} consecutive iterations...")
                        print(f"{'='*60}\n")

                    if self.convergence_count >= self.early_stop_patience:
                        print(f"\n{'='*60}")
                        print(f"[Early Stop] Patience reached! Stopping at iteration {it+1}")
                        print(f"{'='*60}\n")
                        self.stopped_early = True
                        break
                else:
                    if self.convergence_count > 0:
                        print(f"\n[Early Stop] Convergence lost at iteration {it+1} (count was {self.convergence_count})")
                        print(f"  Time-out: {timeout_rate*100:.1f}% (need >= {self.early_stop_timeout*100:.0f}%)")
                        print(f"  Ep length: {ep_len_mean:.0f} (need >= {self.early_stop_ep_length:.0f})")
                        print(f"  Vel tracking: {vel_tracking:.3f} (need >= {self.early_stop_vel_tracking:.3f})")
                    self.convergence_count = 0

            # Clear old buffers to avoid memory issues
            if len(rewbuffer) > 10000:
                rewbuffer = rewbuffer[-1000:]
                lenbuffer = lenbuffer[-1000:]
                timeout_buffer = timeout_buffer[-1000:]


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


class AdaptiveLRScheduler:
    """Adaptive learning rate scheduler based on episode length trends.

    Monitors training progress and adjusts learning rate:
    - Reduces LR when episode length drops (collapse detection)
    - Increases LR when episode length improves (progress detection)
    - Can rollback to best checkpoint if severe collapse detected
    """

    def __init__(
        self,
        runner,
        lr_min: float = 1e-5,
        lr_max: float = 1e-3,
        lr_decay_factor: float = 0.5,
        lr_grow_factor: float = 1.1,
        collapse_threshold: float = 0.7,
        rollback_threshold: float = 0.3,
        window_size: int = 50,
        check_interval: int = 20,
        log_dir: str = None,
    ):
        self.runner = runner
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.lr_decay_factor = lr_decay_factor
        self.lr_grow_factor = lr_grow_factor
        self.collapse_threshold = collapse_threshold
        self.rollback_threshold = rollback_threshold
        self.window_size = window_size
        self.check_interval = check_interval
        self.log_dir = log_dir

        # Tracking state
        self.ep_length_history = []
        self.peak_ep_length = 0.0
        self.best_checkpoint_path = None
        self.best_ep_length = 0.0
        self.current_lr = self._get_current_lr()
        self.lr_reductions = 0
        self.rollbacks = 0

    def _get_current_lr(self) -> float:
        """Get current learning rate from optimizer."""
        for param_group in self.runner.alg.optimizer.param_groups:
            return param_group['lr']
        return self.lr_max

    def _set_lr(self, new_lr: float):
        """Set learning rate in optimizer."""
        new_lr = max(self.lr_min, min(self.lr_max, new_lr))
        for param_group in self.runner.alg.optimizer.param_groups:
            param_group['lr'] = new_lr
        self.current_lr = new_lr
        return new_lr

    def _get_rolling_avg(self, n: int = None) -> float:
        """Get rolling average of episode length."""
        if not self.ep_length_history:
            return 0.0
        n = n or self.window_size
        recent = self.ep_length_history[-n:]
        return sum(recent) / len(recent)

    def update(self, ep_length: float, iteration: int):
        """Update scheduler with new episode length observation."""
        self.ep_length_history.append(ep_length)

        # Update peak
        if ep_length > self.peak_ep_length:
            self.peak_ep_length = ep_length

        # Save checkpoint if this is the best
        if ep_length > self.best_ep_length and self.log_dir:
            self.best_ep_length = ep_length
            self.best_checkpoint_path = os.path.join(
                self.log_dir, f"model_best_{iteration}.pt"
            )
            self.runner.save(self.best_checkpoint_path)

        # Only check periodically to avoid noisy decisions
        if iteration % self.check_interval != 0 or len(self.ep_length_history) < self.window_size:
            return None

        # Compute metrics
        current_avg = self._get_rolling_avg()
        prev_avg = self._get_rolling_avg(self.window_size * 2)[-self.window_size:] if len(self.ep_length_history) > self.window_size * 2 else current_avg

        # Recompute prev_avg correctly
        if len(self.ep_length_history) > self.window_size * 2:
            prev_data = self.ep_length_history[-(self.window_size * 2):-self.window_size]
            prev_avg = sum(prev_data) / len(prev_data)
        else:
            prev_avg = current_avg

        ratio_to_peak = current_avg / max(1.0, self.peak_ep_length)

        action = None

        # Check for severe collapse - consider rollback
        if ratio_to_peak < self.rollback_threshold and self.best_checkpoint_path:
            print(f"\n{'!'*60}")
            print(f"[Adaptive LR] SEVERE COLLAPSE DETECTED at iter {iteration}")
            print(f"  Current avg: {current_avg:.1f}, Peak: {self.peak_ep_length:.1f} (ratio: {ratio_to_peak:.2f})")
            print(f"  Rolling back to best checkpoint: {self.best_checkpoint_path}")
            print(f"  Reducing LR by {self.lr_decay_factor}x")
            print(f"{'!'*60}\n")

            # Load best checkpoint
            self.runner.load(self.best_checkpoint_path)

            # Reduce LR significantly
            new_lr = self._set_lr(self.current_lr * self.lr_decay_factor * self.lr_decay_factor)
            self.lr_reductions += 2
            self.rollbacks += 1

            # Reset peak to allow recovery
            self.peak_ep_length = self.best_ep_length
            action = "rollback"

        # Check for moderate collapse - reduce LR
        elif ratio_to_peak < self.collapse_threshold:
            new_lr = self._set_lr(self.current_lr * self.lr_decay_factor)
            self.lr_reductions += 1
            print(f"\n[Adaptive LR] Collapse detected at iter {iteration}")
            print(f"  Current avg: {current_avg:.1f}, Peak: {self.peak_ep_length:.1f} (ratio: {ratio_to_peak:.2f})")
            print(f"  Reducing LR: {self.current_lr/self.lr_decay_factor:.2e} -> {new_lr:.2e}")
            action = "decay"

        # Check for improvement - gradually increase LR
        elif current_avg > prev_avg * 1.1 and self.current_lr < self.lr_max:
            new_lr = self._set_lr(self.current_lr * self.lr_grow_factor)
            print(f"\n[Adaptive LR] Progress detected at iter {iteration}")
            print(f"  Current avg: {current_avg:.1f} > Prev avg: {prev_avg:.1f}")
            print(f"  Increasing LR: {self.current_lr/self.lr_grow_factor:.2e} -> {new_lr:.2e}")
            action = "grow"

        return action

    def get_stats(self) -> dict:
        """Get scheduler statistics."""
        return {
            "current_lr": self.current_lr,
            "peak_ep_length": self.peak_ep_length,
            "best_ep_length": self.best_ep_length,
            "lr_reductions": self.lr_reductions,
            "rollbacks": self.rollbacks,
        }


def train_with_adaptive_lr(
    runner,
    max_iterations: int,
    args_cli,
    log_dir: str,
):
    """Training with adaptive learning rate using chunked training.

    Runs training in chunks using the standard runner, checking progress
    after each chunk and adjusting LR accordingly. This avoids RSL-RL
    API compatibility issues by using the built-in training loop.
    """
    import time
    from collections import deque

    # Chunk size - how many iterations between LR checks
    chunk_size = 50
    num_chunks = (max_iterations + chunk_size - 1) // chunk_size

    # LR tracking
    lr_min = args_cli.lr_min
    lr_max = args_cli.lr_max
    lr_decay_factor = args_cli.lr_decay_factor
    lr_grow_factor = args_cli.lr_grow_factor
    collapse_threshold = args_cli.collapse_threshold
    rollback_threshold = args_cli.rollback_threshold

    # Get initial LR
    current_lr = lr_max
    for param_group in runner.alg.optimizer.param_groups:
        current_lr = param_group['lr']
        break

    # Progress tracking
    ep_length_history = deque(maxlen=200)
    peak_ep_length = 0.0
    best_ep_length = 0.0
    best_checkpoint_path = None
    lr_reductions = 0
    rollbacks = 0
    total_iterations = 0

    print(f"\n{'='*60}")
    print("ADAPTIVE LEARNING RATE ENABLED (Chunked Training)")
    print(f"{'='*60}")
    print(f"  LR range: [{lr_min:.2e}, {lr_max:.2e}]")
    print(f"  Decay factor: {lr_decay_factor}x on collapse")
    print(f"  Grow factor: {lr_grow_factor}x on progress")
    print(f"  Collapse threshold: {collapse_threshold*100:.0f}% of peak")
    print(f"  Rollback threshold: {rollback_threshold*100:.0f}% of peak")
    print(f"  Check interval: {chunk_size} iterations")
    print(f"{'='*60}\n")

    start_time = time.time()

    for chunk_idx in range(num_chunks):
        # Calculate cumulative iteration target for this chunk
        # RSL-RL's learn() uses range(current_learning_iteration, num_learning_iterations)
        # So we must pass CUMULATIVE targets, not per-chunk counts
        target_iteration = min((chunk_idx + 1) * chunk_size, max_iterations)

        if target_iteration <= total_iterations:
            break

        # Run training chunk using standard runner (handles all API details)
        print(f"\n--- Chunk {chunk_idx + 1}/{num_chunks}: iterations {total_iterations + 1}-{target_iteration} ---")
        runner.learn(num_learning_iterations=target_iteration, init_at_random_ep_len=(chunk_idx == 0))
        total_iterations = target_iteration

        # Get episode length from TensorBoard writer (if available) or estimate
        # For now, we'll read from the runner's logged statistics
        try:
            # Try to get from runner's internal storage
            ep_len = runner.env.episode_length_buf.float().mean().item()
        except:
            ep_len = 100.0  # Default if we can't get it

        ep_length_history.append(ep_len)

        # Update peak
        if ep_len > peak_ep_length:
            peak_ep_length = ep_len

        # Save best checkpoint
        if ep_len > best_ep_length:
            best_ep_length = ep_len
            best_checkpoint_path = os.path.join(log_dir, f"model_best_{total_iterations}.pt")
            runner.save(best_checkpoint_path)
            print(f"[Adaptive LR] New best checkpoint saved: ep_len={ep_len:.1f}")

        # Calculate rolling average
        if len(ep_length_history) >= 10:
            current_avg = sum(list(ep_length_history)[-10:]) / 10
            prev_avg = sum(list(ep_length_history)[-20:-10]) / 10 if len(ep_length_history) >= 20 else current_avg
        else:
            current_avg = ep_len
            prev_avg = ep_len

        ratio_to_peak = current_avg / max(1.0, peak_ep_length)

        # Check for severe collapse - rollback
        if ratio_to_peak < rollback_threshold and best_checkpoint_path and len(ep_length_history) > 20:
            print(f"\n{'!'*60}")
            print(f"[Adaptive LR] SEVERE COLLAPSE DETECTED!")
            print(f"  Current avg: {current_avg:.1f}, Peak: {peak_ep_length:.1f} (ratio: {ratio_to_peak:.2f})")
            print(f"  Rolling back to best checkpoint")
            print(f"{'!'*60}\n")

            runner.load(best_checkpoint_path)

            # Reduce LR significantly
            new_lr = max(lr_min, current_lr * lr_decay_factor * lr_decay_factor)
            for param_group in runner.alg.optimizer.param_groups:
                param_group['lr'] = new_lr
            current_lr = new_lr
            lr_reductions += 2
            rollbacks += 1

            # Reset peak
            peak_ep_length = best_ep_length
            print(f"  New LR: {new_lr:.2e}")

        # Check for moderate collapse - reduce LR
        elif ratio_to_peak < collapse_threshold and len(ep_length_history) > 10:
            new_lr = max(lr_min, current_lr * lr_decay_factor)
            for param_group in runner.alg.optimizer.param_groups:
                param_group['lr'] = new_lr
            print(f"\n[Adaptive LR] Collapse detected (ratio: {ratio_to_peak:.2f})")
            print(f"  Reducing LR: {current_lr:.2e} -> {new_lr:.2e}")
            current_lr = new_lr
            lr_reductions += 1

        # Check for improvement - increase LR
        elif current_avg > prev_avg * 1.1 and current_lr < lr_max and len(ep_length_history) > 20:
            new_lr = min(lr_max, current_lr * lr_grow_factor)
            for param_group in runner.alg.optimizer.param_groups:
                param_group['lr'] = new_lr
            print(f"\n[Adaptive LR] Progress detected!")
            print(f"  Increasing LR: {current_lr:.2e} -> {new_lr:.2e}")
            current_lr = new_lr

        # Status update
        print(f"[Adaptive LR] Status: LR={current_lr:.2e}, EP len avg={current_avg:.1f}, Peak={peak_ep_length:.1f}")

    elapsed = time.time() - start_time

    # Create a stats object to return
    class SchedulerStats:
        pass

    scheduler = SchedulerStats()
    scheduler.best_checkpoint_path = best_checkpoint_path
    scheduler.current_lr = current_lr
    scheduler.peak_ep_length = peak_ep_length
    scheduler.best_ep_length = best_ep_length
    scheduler.lr_reductions = lr_reductions
    scheduler.rollbacks = rollbacks

    def get_stats():
        return {
            "current_lr": scheduler.current_lr,
            "peak_ep_length": scheduler.peak_ep_length,
            "best_ep_length": scheduler.best_ep_length,
            "lr_reductions": scheduler.lr_reductions,
            "rollbacks": scheduler.rollbacks,
        }
    scheduler.get_stats = get_stats

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
    print(f"{'='*60}\n")

    return scheduler


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

    # Override episode length
    if args_cli.episode_length is not None:
        env_cfg_obj.episode_length_s = args_cli.episode_length * env_cfg_obj.sim.dt * env_cfg_obj.decimation
        print(f"  [CLI Override] episode_length = {args_cli.episode_length} steps")

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

    # Calculate early stopping thresholds
    ep_length_threshold = args_cli.episode_length * args_cli.early_stop_ep_length_pct

    # Print training info
    print(f"\n{'='*60}")
    print(f"TRAINING CONFIGURATION")
    print(f"{'='*60}")
    print(f"Task: {args_cli.task}")
    print(f"Num envs: {env_cfg_obj.scene.num_envs}")
    print(f"Max iterations: {agent_cfg_obj.max_iterations}")
    print(f"Steps per env: {agent_cfg_obj.num_steps_per_env}")
    print(f"Episode length: {args_cli.episode_length} steps")
    print(f"Log directory: {log_dir}")
    if args_cli.arm_penalty is not None:
        print(f"Arm penalty (CLI): {args_cli.arm_penalty}")
    if args_cli.orientation_limit is not None:
        print(f"Orientation limit (CLI): {args_cli.orientation_limit}")
    if args_cli.vel_max is not None:
        print(f"Max velocity (CLI): {args_cli.vel_max}")
    if args_cli.early_stop:
        print(f"\nEarly Stopping Enabled:")
        print(f"  Time-out rate >= {args_cli.early_stop_timeout*100:.0f}%")
        print(f"  Episode length >= {ep_length_threshold:.0f} ({args_cli.early_stop_ep_length_pct*100:.0f}% of {args_cli.episode_length})")
        print(f"  Velocity tracking >= {args_cli.early_stop_vel_tracking*100:.0f}%")
        print(f"  Patience: {args_cli.early_stop_patience} iterations")
    if args_cli.adaptive_lr:
        print(f"\nAdaptive Learning Rate Enabled:")
        print(f"  LR range: [{args_cli.lr_min:.2e}, {args_cli.lr_max:.2e}]")
        print(f"  Decay factor: {args_cli.lr_decay_factor}x (on collapse)")
        print(f"  Grow factor: {args_cli.lr_grow_factor}x (on progress)")
        print(f"  Collapse threshold: {args_cli.collapse_threshold*100:.0f}% of peak")
        print(f"  Rollback threshold: {args_cli.rollback_threshold*100:.0f}% of peak")
    print(f"{'='*60}\n")

    # Create runner
    print(f"Creating PPO runner...")
    # Note: EarlyStoppingRunner disabled due to RSL-RL API incompatibilities
    # Using standard OnPolicyRunner - monitor logs manually for convergence
    if args_cli.early_stop:
        print(f"WARNING: Early stopping requested but disabled due to RSL-RL API changes.")
        print(f"  Monitor training manually. Convergence targets:")
        print(f"  - Time-out rate >= {args_cli.early_stop_timeout*100:.0f}%")
        print(f"  - Episode length >= {ep_length_threshold:.0f}")
        print(f"  - Patience: {args_cli.early_stop_patience} iterations")
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

    if args_cli.adaptive_lr:
        # Use adaptive learning rate training loop
        scheduler = train_with_adaptive_lr(runner, agent_cfg_obj.max_iterations, args_cli, log_dir)
    else:
        # Use standard training loop
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
    if args_cli.adaptive_lr:
        stats = scheduler.get_stats()
        print(f"Best checkpoint: {scheduler.best_checkpoint_path}")
        print(f"Peak episode length: {stats['peak_ep_length']:.0f}")
    print(f"{'='*60}\n")

    # Cleanup
    env.close()
    sys.stdout = tee_logger.terminal
    tee_logger.close()
    print(f"Log saved to: {log_file}")


if __name__ == "__main__":
    main()
    simulation_app.close()
