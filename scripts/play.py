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

    # Metric collection for posture analysis
    parser.add_argument(
        "--collect_metrics",
        action="store_true",
        help="Collect detailed posture metrics for analysis",
    )
    parser.add_argument(
        "--metrics_interval",
        type=int,
        default=100,
        help="Print metrics summary every N steps",
    )
    parser.add_argument(
        "--metrics_file",
        type=str,
        default=None,
        help="Save metrics to CSV file",
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
import numpy as np
from collections import defaultdict

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg, load_cfg_from_registry

# Import to register environments
import digit_locomotion  # noqa: F401


class PostureMetricsCollector:
    """Collects and analyzes posture metrics during evaluation."""

    def __init__(self, device: str):
        self.device = device
        self.joint_indices = {}
        self.initialized = False
        self.metrics_history = defaultdict(list)

    def initialize(self, joint_names: list):
        """Find relevant joint indices from joint names."""
        for i, name in enumerate(joint_names):
            name_lower = name.lower()

            # Shoulder joints (arm lateral control)
            if "left" in name_lower and "shoulder" in name_lower and "roll" in name_lower:
                self.joint_indices["left_shoulder_roll"] = i
            elif "right" in name_lower and "shoulder" in name_lower and "roll" in name_lower:
                self.joint_indices["right_shoulder_roll"] = i
            elif "left" in name_lower and "shoulder" in name_lower and "yaw" in name_lower:
                self.joint_indices["left_shoulder_yaw"] = i
            elif "right" in name_lower and "shoulder" in name_lower and "yaw" in name_lower:
                self.joint_indices["right_shoulder_yaw"] = i
            elif "left" in name_lower and "shoulder" in name_lower and "pitch" in name_lower:
                self.joint_indices["left_shoulder_pitch"] = i
            elif "right" in name_lower and "shoulder" in name_lower and "pitch" in name_lower:
                self.joint_indices["right_shoulder_pitch"] = i

            # Elbow joints
            elif "left" in name_lower and "elbow" in name_lower:
                self.joint_indices["left_elbow"] = i
            elif "right" in name_lower and "elbow" in name_lower:
                self.joint_indices["right_elbow"] = i

            # Hip pitch (for arm swing coordination)
            elif "left" in name_lower and "hip" in name_lower and "pitch" in name_lower:
                self.joint_indices["left_hip_pitch"] = i
            elif "right" in name_lower and "hip" in name_lower and "pitch" in name_lower:
                self.joint_indices["right_hip_pitch"] = i

        print(f"\n[PostureMetrics] Found joint indices:")
        for name, idx in sorted(self.joint_indices.items()):
            print(f"  {name}: {idx}")
        self.initialized = True

    def collect(self, robot_data, projected_gravity, cmd_vel):
        """Collect posture metrics for current step."""
        if not self.initialized:
            return {}

        joint_pos = robot_data.joint_pos
        joint_vel = robot_data.joint_vel

        metrics = {}

        # Shoulder roll (lateral arm extension) - should be close to 0
        if "left_shoulder_roll" in self.joint_indices:
            left_roll = joint_pos[:, self.joint_indices["left_shoulder_roll"]]
            right_roll = joint_pos[:, self.joint_indices["right_shoulder_roll"]]
            metrics["left_shoulder_roll"] = left_roll.mean().item()
            metrics["right_shoulder_roll"] = right_roll.mean().item()
            metrics["shoulder_roll_abs_mean"] = (torch.abs(left_roll) + torch.abs(right_roll)).mean().item() / 2

        # Shoulder yaw - should be close to 0
        if "left_shoulder_yaw" in self.joint_indices:
            left_yaw = joint_pos[:, self.joint_indices["left_shoulder_yaw"]]
            right_yaw = joint_pos[:, self.joint_indices["right_shoulder_yaw"]]
            metrics["left_shoulder_yaw"] = left_yaw.mean().item()
            metrics["right_shoulder_yaw"] = right_yaw.mean().item()

        # Shoulder pitch (arm swing)
        if "left_shoulder_pitch" in self.joint_indices:
            left_pitch = joint_pos[:, self.joint_indices["left_shoulder_pitch"]]
            right_pitch = joint_pos[:, self.joint_indices["right_shoulder_pitch"]]
            metrics["left_shoulder_pitch"] = left_pitch.mean().item()
            metrics["right_shoulder_pitch"] = right_pitch.mean().item()

        # Elbow bend - should be ~0.8 rad (~45 deg) for jogging
        if "left_elbow" in self.joint_indices:
            left_elbow = joint_pos[:, self.joint_indices["left_elbow"]]
            right_elbow = joint_pos[:, self.joint_indices["right_elbow"]]
            metrics["left_elbow"] = left_elbow.mean().item()
            metrics["right_elbow"] = right_elbow.mean().item()

        # Forward lean (from projected gravity x component)
        # Positive = leaning forward
        metrics["forward_lean"] = projected_gravity[:, 0].mean().item()
        metrics["lateral_tilt"] = projected_gravity[:, 1].mean().item()

        # Arm swing coordination (velocity correlation)
        if "left_hip_pitch" in self.joint_indices and "left_shoulder_pitch" in self.joint_indices:
            left_hip_vel = joint_vel[:, self.joint_indices["left_hip_pitch"]]
            right_hip_vel = joint_vel[:, self.joint_indices["right_hip_pitch"]]
            left_arm_vel = joint_vel[:, self.joint_indices["left_shoulder_pitch"]]
            right_arm_vel = joint_vel[:, self.joint_indices["right_shoulder_pitch"]]

            # Positive = proper coordination (hip and opposite arm same direction)
            coord_left = (left_hip_vel * right_arm_vel).mean().item()
            coord_right = (right_hip_vel * left_arm_vel).mean().item()
            metrics["arm_swing_coordination"] = coord_left + coord_right

        # Commanded velocity
        metrics["cmd_vel_x"] = cmd_vel[:, 0].mean().item()
        metrics["cmd_vel_y"] = cmd_vel[:, 1].mean().item()

        # Store in history
        for key, value in metrics.items():
            self.metrics_history[key].append(value)

        return metrics

    def get_summary(self):
        """Get summary statistics of collected metrics."""
        summary = {}
        for key, values in self.metrics_history.items():
            arr = np.array(values)
            summary[key] = {
                "mean": np.mean(arr),
                "std": np.std(arr),
                "min": np.min(arr),
                "max": np.max(arr),
            }
        return summary

    def print_summary(self, title="Posture Metrics Summary"):
        """Print formatted summary of metrics."""
        summary = self.get_summary()
        if not summary:
            print("No metrics collected yet")
            return

        print(f"\n{'='*70}")
        print(f" {title}")
        print(f"{'='*70}")

        # Group metrics for better readability
        groups = {
            "Shoulder Roll (lateral arm extension, target: 0)": [
                "left_shoulder_roll", "right_shoulder_roll", "shoulder_roll_abs_mean"
            ],
            "Shoulder Yaw (target: 0)": [
                "left_shoulder_yaw", "right_shoulder_yaw"
            ],
            "Shoulder Pitch (arm swing)": [
                "left_shoulder_pitch", "right_shoulder_pitch"
            ],
            "Elbow Bend (target: ~0.8 rad for jogging)": [
                "left_elbow", "right_elbow"
            ],
            "Body Orientation": [
                "forward_lean", "lateral_tilt"
            ],
            "Arm Swing Coordination (target: positive)": [
                "arm_swing_coordination"
            ],
            "Commanded Velocity": [
                "cmd_vel_x", "cmd_vel_y"
            ],
        }

        for group_name, keys in groups.items():
            print(f"\n{group_name}:")
            for key in keys:
                if key in summary:
                    s = summary[key]
                    print(f"  {key:30s}: mean={s['mean']:7.3f}, std={s['std']:6.3f}, "
                          f"range=[{s['min']:7.3f}, {s['max']:7.3f}]")

        print(f"\n{'='*70}")

        # Diagnosis
        print("\nDIAGNOSIS:")
        issues = []

        # Check shoulder roll (lateral extension)
        if "shoulder_roll_abs_mean" in summary:
            roll = summary["shoulder_roll_abs_mean"]["mean"]
            if roll > 0.3:
                issues.append(f"  - Arms extending sideways: shoulder_roll_abs_mean = {roll:.3f} (should be < 0.3)")

        # Check elbow bend
        if "left_elbow" in summary:
            left_e = abs(summary["left_elbow"]["mean"])
            right_e = abs(summary["right_elbow"]["mean"])
            if left_e < 0.5 or right_e < 0.5:
                issues.append(f"  - Arms too straight: elbows = {left_e:.3f}, {right_e:.3f} (target: ~0.8)")

        # Check arm swing
        if "arm_swing_coordination" in summary:
            coord = summary["arm_swing_coordination"]["mean"]
            if coord < 0.1:
                issues.append(f"  - Poor arm swing coordination: {coord:.3f} (should be positive)")

        if issues:
            for issue in issues:
                print(issue)
        else:
            print("  No major posture issues detected!")

        print()

    def save_to_csv(self, filepath):
        """Save metrics history to CSV file."""
        import csv

        if not self.metrics_history:
            print("No metrics to save")
            return

        keys = list(self.metrics_history.keys())
        n_samples = len(self.metrics_history[keys[0]])

        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['step'] + keys)
            writer.writeheader()
            for i in range(n_samples):
                row = {'step': i}
                for key in keys:
                    row[key] = self.metrics_history[key][i]
                writer.writerow(row)

        print(f"Metrics saved to: {filepath}")


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
    if args.collect_metrics:
        print(f"Collecting posture metrics (summary every {args.metrics_interval} steps)")
        if args.metrics_file:
            print(f"Saving metrics to: {args.metrics_file}")
    print("=" * 60)
    print("\nRunning policy... Press Ctrl+C to stop\n")

    # Initialize posture metrics collector if requested
    metrics_collector = None
    if args.collect_metrics:
        metrics_collector = PostureMetricsCollector(device=env.device)
        # Initialize with joint names from robot
        base_env = env.unwrapped
        robot = base_env.scene["robot"]
        metrics_collector.initialize(robot.data.joint_names)

    # Get initial observations
    obs = env.get_observations()

    # Evaluation loop
    step = 0
    episode_rewards = torch.zeros(env.num_envs, device=env.device)
    episode_lengths = torch.zeros(env.num_envs, device=env.device)

    try:
        while simulation_app.is_running():
            # Access the underlying Isaac Lab environment
            base_env = env.unwrapped

            # Override velocity commands if specified
            if use_fixed_vel:
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

            # Collect posture metrics
            if metrics_collector is not None:
                robot = base_env.scene["robot"]
                cmd_vel = base_env.command_manager.get_command("base_velocity")
                metrics_collector.collect(
                    robot.data,
                    robot.data.projected_gravity_b,
                    cmd_vel
                )

            # Track metrics
            episode_rewards += rewards
            episode_lengths += 1

            # Print stats periodically
            if step % 50 == 0 and not args.collect_metrics:
                print(f"Step {step}: Mean reward = {rewards.mean().item():.4f}")

            # Print metrics summary periodically
            if args.collect_metrics and step > 0 and step % args.metrics_interval == 0:
                metrics_collector.print_summary(f"Step {step} Posture Metrics")

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

        # Print final metrics summary
        if metrics_collector is not None:
            metrics_collector.print_summary("FINAL Posture Metrics Summary")
            if args.metrics_file:
                metrics_collector.save_to_csv(args.metrics_file)

    # Print final metrics if not already printed (from Ctrl+C)
    if metrics_collector is not None and step > 0:
        metrics_collector.print_summary("FINAL Posture Metrics Summary")
        if args.metrics_file:
            metrics_collector.save_to_csv(args.metrics_file)

    # Cleanup
    print(f"\nTotal steps: {step}")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
