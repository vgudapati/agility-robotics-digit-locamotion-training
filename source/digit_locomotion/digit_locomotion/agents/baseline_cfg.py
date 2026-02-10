"""RSL-RL configurations for baseline training (Radosavovic et al. 2024).

This module provides configurations for:
1. Teacher policy (MLP with privileged state) - fast training
2. Student policy (Transformer with noisy obs) - final deployment

Training pipeline:
1. Train teacher with DigitBaselineTeacherPPORunnerCfg
2. Distill teacher to student with DigitBaselineStudentPPORunnerCfg
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoActorCriticRecurrentCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class RslRlPpoActorCriticTransformerCfg:
    """Configuration for transformer-based actor-critic.

    Matches the architecture from Radosavovic et al. (2024):
    - 4 transformer blocks
    - 192 embedding dimension
    - 4 attention heads
    - 16 timestep context window
    - ~1.4M parameters
    """

    class_name: str = "ActorCriticTransformer"

    # Observation normalization
    actor_obs_normalization: bool = False
    critic_obs_normalization: bool = False

    # Transformer architecture (paper: 4 blocks, 192 dim, 4 heads)
    transformer_embed_dim: int = 192
    transformer_num_heads: int = 4
    transformer_num_layers: int = 4
    transformer_mlp_ratio: float = 2.0
    transformer_dropout: float = 0.0

    # Context length (paper: 16 timesteps)
    context_length: int = 16

    # Action head (paper: [256, 128])
    actor_hidden_dims: list[int] = [256, 128]
    critic_hidden_dims: list[int] = [256, 128]
    activation: str = "elu"

    # Action noise
    init_noise_std: float = 1.0
    noise_std_type: str = "scalar"
    state_dependent_std: bool = False


@configclass
class DigitBaselineTeacherPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO configuration for TEACHER policy.

    Teacher uses MLP with privileged state (no noise) for fast training.
    Paper uses [512, 512, 256, 128] for teacher state model.

    Training tip: Train teacher first to convergence (~5-10k iterations),
    then use as supervision for student transformer.
    """

    seed = 42  # Fixed seed for reproducibility
    num_steps_per_env = 48
    max_iterations = 10000
    save_interval = 50
    experiment_name = "digit_baseline_teacher"
    run_name = ""
    logger = "tensorboard"
    neptune_project = ""
    wandb_project = ""
    resume = False
    empirical_normalization = False

    # Teacher MLP (privileged state, no noise)
    # Paper: [512, 512, 256, 128]
    policy: RslRlPpoActorCriticCfg = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 512, 256, 128],
        critic_hidden_dims=[512, 512, 256, 128],
        activation="elu",
    )

    algorithm: RslRlPpoAlgorithmCfg = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=8,
        learning_rate=3.0e-4,  # Lower initial LR to prevent spike
        schedule="adaptive",
        desired_kl=0.01,
        max_grad_norm=1.0,
        gamma=0.99,
        lam=0.95,
    )


@configclass
class DigitBaselineTeacherCommunityPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Community-standard PPO configuration for TEACHER policy.

    Tuned for faster convergence based on legged_gym/IsaacLab community defaults:
    - num_steps_per_env: 48 -> 24 (halves collection time, 2x faster iterations)
    - num_mini_batches: 8 -> 4 (larger batches = more stable gradients)
    - num_learning_epochs: 5 -> 8 (more gradient updates per rollout)
    - init_noise_std: 1.0 -> 0.8 (less chaotic for high-DOF humanoid)
    - gamma: 0.99 -> 0.97 (shorter horizon for early training survival)
    - Network: [512,256,128] (3 layers, matches AnymalB standard)

    Sources: legged_gym defaults, IsaacLab AnymalB config, Unitree G1 config.
    """

    seed = 42
    num_steps_per_env = 24  # Community standard (was 48)
    max_iterations = 10000
    save_interval = 50
    experiment_name = "digit_baseline_teacher_community"
    run_name = ""
    logger = "tensorboard"
    neptune_project = ""
    wandb_project = ""
    resume = False
    empirical_normalization = False

    policy: RslRlPpoActorCriticCfg = RslRlPpoActorCriticCfg(
        init_noise_std=0.8,  # Reduced for humanoid (Unitree G1 standard)
        actor_hidden_dims=[512, 256, 128],  # 3-layer (AnymalB standard)
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm: RslRlPpoAlgorithmCfg = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.02,  # Increased to push past plateau (was 0.01)
        num_learning_epochs=8,  # More gradient steps (was 5)
        num_mini_batches=4,  # Larger batches (was 8)
        learning_rate=3.0e-4,
        schedule="adaptive",
        desired_kl=0.01,
        max_grad_norm=1.0,
        gamma=0.99,  # Restored: longer horizon now that robot survives 72+ steps
        lam=0.95,
    )


@configclass
class DigitBaselineStudentPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO configuration for STUDENT transformer policy.

    Student uses transformer with observation-action history.
    Trained with joint objective: RL + KL divergence from teacher.

    Paper: "λ is gradually annealed to zero over the course of the training
    process, typically reaching zero at the midpoint of the training horizon."
    """

    seed = 42  # Fixed seed for reproducibility
    num_steps_per_env = 48
    max_iterations = 20000  # Longer training for transformer
    save_interval = 50
    experiment_name = "digit_baseline_student"
    run_name = ""
    logger = "tensorboard"
    neptune_project = ""
    wandb_project = ""
    resume = False
    empirical_normalization = False

    # Transformer policy (paper architecture)
    policy: RslRlPpoActorCriticTransformerCfg = RslRlPpoActorCriticTransformerCfg()

    algorithm: RslRlPpoAlgorithmCfg = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=8,
        learning_rate=5.0e-4,  # Lower LR for transformer stability
        schedule="adaptive",
        desired_kl=0.01,
        max_grad_norm=1.0,
        gamma=0.99,
        lam=0.95,
    )


@configclass
class DigitBaselineMlpPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Alternative: MLP baseline for comparison.

    Same rewards as baseline but with standard MLP policy.
    Use this to measure the benefit of transformer architecture.
    """

    seed = 42  # Fixed seed for reproducibility
    num_steps_per_env = 48
    max_iterations = 15000
    save_interval = 50
    experiment_name = "digit_baseline_mlp"
    run_name = ""
    logger = "tensorboard"
    neptune_project = ""
    wandb_project = ""
    resume = False
    empirical_normalization = False

    policy: RslRlPpoActorCriticCfg = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[1024, 512, 256],
        critic_hidden_dims=[1024, 512, 256],
        activation="elu",
    )

    algorithm: RslRlPpoAlgorithmCfg = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=8,
        learning_rate=3.0e-4,  # Lower initial LR to prevent spike
        schedule="adaptive",
        desired_kl=0.01,
        max_grad_norm=1.0,
        gamma=0.99,
        lam=0.95,
    )


@configclass
class DigitBaselineRecurrentPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Alternative: LSTM baseline for comparison.

    Uses recurrent policy (LSTM) instead of transformer.
    Paper shows transformer outperforms LSTM by significant margin.
    """

    seed = 42  # Fixed seed for reproducibility
    num_steps_per_env = 48
    max_iterations = 15000
    save_interval = 50
    experiment_name = "digit_baseline_lstm"
    run_name = ""
    logger = "tensorboard"
    neptune_project = ""
    wandb_project = ""
    resume = False
    empirical_normalization = False

    policy: RslRlPpoActorCriticRecurrentCfg = RslRlPpoActorCriticRecurrentCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 256],
        critic_hidden_dims=[256, 256],
        activation="elu",
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
    )

    algorithm: RslRlPpoAlgorithmCfg = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=8,
        learning_rate=3.0e-4,  # Lower initial LR to prevent spike
        schedule="adaptive",
        desired_kl=0.01,
        max_grad_norm=1.0,
        gamma=0.99,
        lam=0.95,
    )
