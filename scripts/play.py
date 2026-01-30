#!/usr/bin/env python3
"""Evaluation/playback script for trained Digit locomotion policies.

This script loads a trained policy and runs it in the simulation
environment for visualization and evaluation.

Usage (Linux/macOS):
    # Play with a trained checkpoint
    ./isaaclab.sh -p scripts/play.py --task Digit-Velocity-Flat-v0 \
        --checkpoint logs/digit_flat/model_10000.pt

    # Play with visualization (not headless)
    ./isaaclab.sh -p scripts/play.py --task Digit-Velocity-Flat-v0 \
        --checkpoint logs/digit_flat/model_10000.pt --num_envs 16

    # Record video
    ./isaaclab.sh -p scripts/play.py --task Digit-Velocity-Flat-v0 \
        --checkpoint logs/digit_flat/model_10000.pt --video --video_length 300

Usage (Windows):
    # Play with a trained checkpoint
    isaaclab.bat -p scripts\\play.py --task Digit-Velocity-Flat-v0 ^
        --checkpoint logs\\digit_flat\\model_10000.pt

    # Play with visualization
    isaaclab.bat -p scripts\\play.py --task Digit-Velocity-Flat-v0 ^
        --checkpoint logs\\digit_flat\\model_10000.pt --num_envs 16
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
        required=True,
        help="Path to the trained policy checkpoint",
    )

    # Video recording
    parser.add_argument(
        "--video",
        action="store_true",
        help="Record video of evaluation",
    )
    parser.add_argument(
        "--video_length",
        type=int,
        default=200,
        help="Length of video in steps",
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        default="videos",
        help="Directory to save videos",
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

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

# Import to register environments
import digit_locomotion  # noqa: F401


def main():
    """Main evaluation function."""
    # Set random seed
    torch.manual_seed(args.seed)

    # Create environment
    env = gym.make(
        args.task,
        cfg={"num_envs": args.num_envs},
    )
    env = RslRlVecEnvWrapper(env)

    # Load the trained policy
    print(f"\nLoading policy from: {args.checkpoint}")

    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    # Load as JIT model or state dict
    if args.checkpoint.endswith(".pt"):
        policy = torch.jit.load(args.checkpoint, map_location=env.device)
    else:
        # Assume it's a state dict - need to reconstruct the model
        from rsl_rl.modules import ActorCritic

        # Get observation and action dimensions
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]

        # Create model with same architecture as training
        policy = ActorCritic(
            num_actor_obs=obs_dim,
            num_critic_obs=obs_dim,
            num_actions=act_dim,
            actor_hidden_dims=[512, 256, 128],
            critic_hidden_dims=[512, 256, 128],
            activation="elu",
        ).to(env.device)

        # Load weights
        checkpoint = torch.load(args.checkpoint, map_location=env.device)
        policy.load_state_dict(checkpoint["model_state_dict"])

    policy.eval()

    # Print info
    print("\n" + "=" * 60)
    print("DIGIT LOCOMOTION EVALUATION")
    print("=" * 60)
    print(f"Task: {args.task}")
    print(f"Number of environments: {env.num_envs}")
    print(f"Checkpoint: {args.checkpoint}")
    print("=" * 60 + "\n")

    # Setup video recording if requested
    video_writer = None
    if args.video:
        os.makedirs(args.video_dir, exist_ok=True)
        video_path = os.path.join(
            args.video_dir,
            f"digit_eval_{os.path.basename(args.checkpoint).split('.')[0]}.mp4"
        )
        print(f"Recording video to: {video_path}")

    # Reset environment
    obs, _ = env.reset()

    # Evaluation loop
    step = 0
    episode_rewards = torch.zeros(env.num_envs, device=env.device)
    episode_lengths = torch.zeros(env.num_envs, device=env.device)

    print("Starting evaluation...")
    print("Press Ctrl+C to stop\n")

    try:
        while simulation_app.is_running():
            # Get action from policy
            with torch.no_grad():
                actions = policy.act(obs)

            # Step environment
            obs, rewards, dones, truncated, infos = env.step(actions)

            # Track metrics
            episode_rewards += rewards
            episode_lengths += 1

            # Print stats on episode end
            done_envs = dones.nonzero(as_tuple=False).squeeze(-1)
            if len(done_envs) > 0:
                mean_reward = episode_rewards[done_envs].mean().item()
                mean_length = episode_lengths[done_envs].mean().item()
                print(f"Step {step}: Completed episodes - "
                      f"Mean reward: {mean_reward:.2f}, "
                      f"Mean length: {mean_length:.0f}")

                # Reset tracked metrics for completed episodes
                episode_rewards[done_envs] = 0
                episode_lengths[done_envs] = 0

            step += 1

            # Check if we should stop for video recording
            if args.video and step >= args.video_length:
                print(f"\nReached video length ({args.video_length} steps)")
                break

    except KeyboardInterrupt:
        print("\nEvaluation stopped by user")

    # Cleanup
    print("\nCleaning up...")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
