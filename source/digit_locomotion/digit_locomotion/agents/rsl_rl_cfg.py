"""RSL-RL PPO agent configurations for Digit locomotion training.

RSL-RL is a high-performance RL library optimized for robot locomotion,
developed by the Robotic Systems Lab at ETH Zurich.

Configurations provided:
1. DigitFlatPPORunnerCfg - For training on flat terrain
2. DigitRoughPPORunnerCfg - For training on rough terrain with more iterations
3. DigitMinimalPPORunnerCfg - Fast experiments with reduced DOF (8 joints)
4. DigitRunningPPORunnerCfg - For high-speed running (up to 5 m/s)
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class DigitFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO configuration for Digit locomotion on flat terrain.

    This configuration is optimized for learning basic walking behavior.
    Training typically converges within 10-15k iterations.

    GPU Optimization Notes (RTX 4090 with 24GB VRAM):
    - Use --num_envs 8192 or 16384 for better GPU utilization
    - num_steps_per_env=48 collects more data per update
    - Larger network [1024, 512, 256] improves learning capacity
    - num_mini_batches=8 for larger effective batch size
    """

    # Runner settings
    # NOTE: Increase num_steps_per_env for better GPU utilization
    num_steps_per_env = 48  # Steps collected per environment before update (was 24)
    max_iterations = 15000  # Total training iterations
    save_interval = 500     # Save checkpoint every N iterations
    experiment_name = "digit_flat"
    run_name = ""           # Auto-generated if empty
    logger = "tensorboard"
    neptune_project = ""
    wandb_project = ""
    resume = False

    # Empirical normalization of observations
    empirical_normalization = False

    # Policy network configuration
    # NOTE: Larger network for better learning capacity and GPU utilization
    policy: RslRlPpoActorCriticCfg = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[1024, 512, 256],   # Larger (was [512, 256, 128])
        critic_hidden_dims=[1024, 512, 256],  # Larger (was [512, 256, 128])
        activation="elu",
    )

    # PPO algorithm configuration
    algorithm: RslRlPpoAlgorithmCfg = RslRlPpoAlgorithmCfg(
        # Value function
        value_loss_coef=1.0,
        use_clipped_value_loss=True,

        # PPO clipping
        clip_param=0.2,

        # Entropy for exploration
        entropy_coef=0.01,

        # Optimization
        num_learning_epochs=5,
        num_mini_batches=8,   # More mini-batches (was 4)
        learning_rate=1.0e-3,
        schedule="adaptive",  # Adaptive LR based on KL divergence
        desired_kl=0.01,
        max_grad_norm=1.0,

        # GAE (Generalized Advantage Estimation)
        gamma=0.99,
        lam=0.95,
    )


@configclass
class DigitRoughPPORunnerCfg(DigitFlatPPORunnerCfg):
    """PPO configuration for Digit locomotion on rough terrain.

    Extended training with more iterations for robust policies.
    Includes terrain curriculum and domain randomization.
    """

    max_iterations = 30000  # More iterations for complex terrain
    experiment_name = "digit_rough"

    # Slightly larger batch for stability with domain randomization
    algorithm: RslRlPpoAlgorithmCfg = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,  # Slightly lower entropy for fine-tuning
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-4,  # Lower LR for stability
        schedule="adaptive",
        desired_kl=0.01,
        max_grad_norm=1.0,
        gamma=0.99,
        lam=0.95,
    )


@configclass
class DigitCurriculumPPORunnerCfg(DigitRoughPPORunnerCfg):
    """PPO configuration with extended curriculum training.

    For training policies that can handle the full range of terrains
    and disturbances. Recommended for sim-to-real transfer.
    """

    max_iterations = 50000
    experiment_name = "digit_curriculum"

    # More conservative learning for stability
    algorithm: RslRlPpoAlgorithmCfg = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=8,  # Larger mini-batches
        learning_rate=3.0e-4,
        schedule="adaptive",
        desired_kl=0.008,
        max_grad_norm=1.0,
        gamma=0.99,
        lam=0.95,
    )


@configclass
class DigitMinimalPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO configuration for minimal Digit (legs only, 8 DOF).

    Optimized for FAST iteration speed:
    - 16384 envs (balanced GPU utilization vs iteration speed)
    - 16 steps per env (fast iterations, more policy updates)
    - Small network (fast forward/backward)

    Speed optimization:
    - Low num_steps_per_env = faster iterations
    - High num_envs = GPU utilization
    - Small network = fast inference
    """

    # Runner settings - optimized for FAST iterations
    num_steps_per_env = 16  # Low = fast iterations (was 48)
    max_iterations = 5000   # Quick experiments (override with --max_iterations)
    save_interval = 50      # Save checkpoints frequently for analysis
    experiment_name = "digit_minimal"
    run_name = ""
    logger = "tensorboard"
    neptune_project = ""
    wandb_project = ""
    resume = False

    empirical_normalization = False

    # Small network for fast inference
    policy: RslRlPpoActorCriticCfg = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 128, 64],   # Small network
        critic_hidden_dims=[256, 128, 64],  # Small network
        activation="elu",
    )

    # PPO algorithm - fast updates
    algorithm: RslRlPpoAlgorithmCfg = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=4,    # Fewer epochs per update (was 5)
        num_mini_batches=4,       # Fewer mini-batches (was 8)
        learning_rate=3.0e-3,     # Higher LR for faster learning
        schedule="adaptive",
        desired_kl=0.01,
        max_grad_norm=1.0,
        gamma=0.99,
        lam=0.95,
    )


@configclass
class DigitRunningPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO configuration for Digit high-speed running (target: 30 mph / 13.4 m/s).

    CURRICULUM TRAINING - Update velocity_env_cfg.py after each phase:
    ================================================================
    Phase 1: lin_vel_x=(0.0, 2.0)   Walking         1000 iter (adjust as needed)
    Phase 2: lin_vel_x=(0.0, 5.0)   Jogging         TBD iter
    Phase 3: lin_vel_x=(0.0, 8.0)   Fast running    TBD iter
    Phase 4: lin_vel_x=(0.0, 10.0)  Sprint warmup   TBD iter
    Phase 5: lin_vel_x=(0.0, 13.5)  30 mph sprint   TBD iter
    ================================================================

    After each phase:
    1. Update lin_vel_x in CommandsRunningCfg (velocity_env_cfg.py)
    2. Resume training: --checkpoint <last_model.pt>

    Configuration:
    - Large network [1024, 512, 256] for complex sprint dynamics
    - 16384 envs for faster training
    - 1000 iterations per phase (quick iteration to find optimal counts)
    """

    # Runner settings
    num_steps_per_env = 24  # Balanced for running
    max_iterations = 1000   # Quick iteration loop (adjust per phase as needed)
    save_interval = 50      # Frequent checkpoints for analysis
    experiment_name = "digit_running"
    run_name = ""
    logger = "tensorboard"
    neptune_project = ""
    wandb_project = ""
    resume = False

    empirical_normalization = False

    # Large network for high-speed sprint dynamics
    policy: RslRlPpoActorCriticCfg = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[1024, 512, 256],   # Large network for 30 mph
        critic_hidden_dims=[1024, 512, 256],  # Large network for 30 mph
        activation="elu",
    )

    # PPO algorithm for running
    algorithm: RslRlPpoAlgorithmCfg = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,     # Standard LR
        schedule="adaptive",
        desired_kl=0.01,
        max_grad_norm=1.0,
        gamma=0.99,
        lam=0.95,
    )
