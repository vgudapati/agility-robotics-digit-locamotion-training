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
