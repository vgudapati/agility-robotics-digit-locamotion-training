"""Causal Transformer policy for humanoid locomotion.

Based on "Real-world humanoid locomotion with reinforcement learning"
(Radosavovic et al., Science Robotics 2024)

Key architecture:
- Causal transformer that takes observation-action history as input
- Context window of 16 timesteps
- 4 transformer blocks, 192 embedding dim, 4 attention heads
- ~1.4M parameters

The transformer processes (obs, action) pairs as tokens and predicts the next action.
This enables in-context adaptation to terrain and disturbances.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal
from typing import Any, NoReturn

from rsl_rl.networks import MLP, EmpiricalNormalization


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""

    def __init__(self, d_model: int, max_len: int = 64, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)

        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class CausalTransformer(nn.Module):
    """Causal Transformer encoder for processing observation-action history.

    Architecture from the paper:
    - 4 transformer blocks
    - 192 embedding dimension
    - 4 attention heads
    - MLP ratio of 2.0
    - Causal attention mask (only attend to past)

    The transformer takes embedded (obs, action) pairs and produces
    contextualized representations for action prediction.
    """

    def __init__(
        self,
        input_dim: int,
        embed_dim: int = 192,
        num_heads: int = 4,
        num_layers: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
        max_context_len: int = 32,
    ):
        """Initialize CausalTransformer.

        Args:
            input_dim: Dimension of input (obs + action concatenated)
            embed_dim: Transformer embedding dimension (paper: 192)
            num_heads: Number of attention heads (paper: 4)
            num_layers: Number of transformer blocks (paper: 4)
            mlp_ratio: MLP hidden dim ratio (paper: 2.0)
            dropout: Dropout rate
            max_context_len: Maximum context window length
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.max_context_len = max_context_len

        # Input projection (paper: MLP with [512, 512])
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ELU(),
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Linear(512, embed_dim),
        )

        # Positional encoding
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=max_context_len, dropout=dropout)

        # Transformer encoder layers with causal masking
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-norm for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Final layer norm
        self.final_norm = nn.LayerNorm(embed_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values for stability."""
        for name, p in self.named_parameters():
            if 'weight' in name and p.dim() >= 2:
                nn.init.xavier_uniform_(p, gain=0.1)
            elif 'bias' in name:
                nn.init.zeros_(p)

    def _generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generate causal attention mask.

        Args:
            seq_len: Sequence length
            device: Device for tensor

        Returns:
            Causal mask where True indicates positions to mask (not attend to)
        """
        # Create upper triangular mask (True = masked, can't attend to future)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through transformer.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
               Contains concatenated (obs, prev_action) pairs

        Returns:
            Output tensor of shape (batch, seq_len, embed_dim)
        """
        batch_size, seq_len, _ = x.shape

        # Project input to embedding dimension
        x = self.input_proj(x)  # (batch, seq_len, embed_dim)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Generate causal mask
        causal_mask = self._generate_causal_mask(seq_len, x.device)

        # Transformer forward pass with causal masking
        x = self.transformer(x, mask=causal_mask, is_causal=True)

        # Final layer norm
        x = self.final_norm(x)

        return x


class ActorCriticTransformer(nn.Module):
    """Transformer-based Actor-Critic for humanoid locomotion.

    Following the architecture from Radosavovic et al. (Science Robotics 2024):
    - Causal transformer processes observation-action history
    - Context window of 16 timesteps for in-context adaptation
    - Separate action head outputs action distribution
    - Critic shares transformer backbone

    Key differences from MLP policies:
    - Uses temporal context for implicit system identification
    - Can adapt behavior based on terrain/disturbance history
    - Enables emergent behaviors like arm swing without explicit rewards
    """

    is_recurrent: bool = True  # Handles sequences like recurrent networks

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        # Observation normalization
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        # Transformer architecture (paper defaults)
        transformer_embed_dim: int = 192,
        transformer_num_heads: int = 4,
        transformer_num_layers: int = 4,
        transformer_mlp_ratio: float = 2.0,
        transformer_dropout: float = 0.0,
        # Context settings
        context_length: int = 16,
        # Action head (paper: MLP [256, 128])
        actor_hidden_dims: tuple[int] | list[int] = [256, 128],
        critic_hidden_dims: tuple[int] | list[int] = [256, 128],
        activation: str = "elu",
        # Action noise
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        state_dependent_std: bool = False,
        **kwargs: dict[str, Any],
    ) -> None:
        """Initialize ActorCriticTransformer.

        Args:
            obs: Sample observation TensorDict for dimension inference
            obs_groups: Mapping of observation groups to keys
            num_actions: Action dimension
            actor_obs_normalization: Normalize actor observations
            critic_obs_normalization: Normalize critic observations
            transformer_embed_dim: Transformer embedding dimension (paper: 192)
            transformer_num_heads: Number of attention heads (paper: 4)
            transformer_num_layers: Number of transformer layers (paper: 4)
            transformer_mlp_ratio: MLP hidden ratio (paper: 2.0)
            transformer_dropout: Dropout rate
            context_length: History context window (paper: 16)
            actor_hidden_dims: Actor MLP dimensions (paper: [256, 128])
            critic_hidden_dims: Critic MLP dimensions
            activation: MLP activation function
            init_noise_std: Initial action noise std
            noise_std_type: How to parameterize noise ('scalar' or 'log')
            state_dependent_std: Whether noise depends on state
        """
        if kwargs:
            print(f"ActorCriticTransformer.__init__ got unexpected arguments: {list(kwargs.keys())}")
        super().__init__()

        # Store configuration
        self.obs_groups = obs_groups
        self.context_length = context_length
        self.state_dependent_std = state_dependent_std
        self.noise_std_type = noise_std_type

        # Calculate observation dimensions
        num_actor_obs = 0
        for obs_group in obs_groups["policy"]:
            assert len(obs[obs_group].shape) == 2, "ActorCriticTransformer only supports 1D observations."
            num_actor_obs += obs[obs_group].shape[-1]
        num_critic_obs = 0
        for obs_group in obs_groups["critic"]:
            assert len(obs[obs_group].shape) == 2, "ActorCriticTransformer only supports 1D observations."
            num_critic_obs += obs[obs_group].shape[-1]

        self.num_actor_obs = num_actor_obs
        self.num_critic_obs = num_critic_obs
        self.num_actions = num_actions

        # Input dimension: obs + previous action
        transformer_input_dim = num_actor_obs + num_actions

        # Observation normalization
        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs)
        else:
            self.actor_obs_normalizer = nn.Identity()

        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs)
        else:
            self.critic_obs_normalizer = nn.Identity()

        # Causal Transformer backbone
        self.transformer = CausalTransformer(
            input_dim=transformer_input_dim,
            embed_dim=transformer_embed_dim,
            num_heads=transformer_num_heads,
            num_layers=transformer_num_layers,
            mlp_ratio=transformer_mlp_ratio,
            dropout=transformer_dropout,
            max_context_len=context_length + 1,
        )
        print(f"Transformer: embed_dim={transformer_embed_dim}, heads={transformer_num_heads}, "
              f"layers={transformer_num_layers}, context={context_length}")

        # Actor head (predicts action distribution)
        if self.state_dependent_std:
            self.actor = MLP(transformer_embed_dim, [2, num_actions], actor_hidden_dims, activation)
        else:
            self.actor = MLP(transformer_embed_dim, num_actions, actor_hidden_dims, activation)
        print(f"Actor MLP: {self.actor}")

        # Critic head (predicts value)
        self.critic = MLP(transformer_embed_dim, 1, critic_hidden_dims, activation)
        print(f"Critic MLP: {self.critic}")

        # Action noise
        if self.state_dependent_std:
            nn.init.zeros_(self.actor[-2].weight[num_actions:])
            if self.noise_std_type == "scalar":
                nn.init.constant_(self.actor[-2].bias[num_actions:], init_noise_std)
            elif self.noise_std_type == "log":
                nn.init.constant_(self.actor[-2].bias[num_actions:],
                                  math.log(init_noise_std + 1e-7))
        else:
            if self.noise_std_type == "scalar":
                self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
            elif self.noise_std_type == "log":
                self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))

        # History buffers (managed during rollout)
        # Shape: (num_envs, context_length, obs_dim + action_dim)
        self._obs_history = None
        self._action_history = None
        self._history_len = None  # Tracks how many valid entries in history

        # Action distribution
        self.distribution = None
        Normal.set_default_validate_args(False)

        # Print parameter count
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")

    def _init_history(self, num_envs: int, device: torch.device) -> None:
        """Initialize history buffers for a new rollout."""
        self._obs_history = torch.zeros(
            num_envs, self.context_length, self.num_actor_obs,
            device=device
        )
        self._action_history = torch.zeros(
            num_envs, self.context_length, self.num_actions,
            device=device
        )
        self._history_len = torch.zeros(num_envs, dtype=torch.long, device=device)

    def _update_history(self, obs: torch.Tensor, action: torch.Tensor) -> None:
        """Update history buffers with new observation-action pair.

        Uses a rolling window - shifts history left and adds new entry at end.
        """
        # Shift history left by 1
        self._obs_history = torch.roll(self._obs_history, shifts=-1, dims=1)
        self._action_history = torch.roll(self._action_history, shifts=-1, dims=1)

        # Add new entry at the end
        self._obs_history[:, -1, :] = obs
        self._action_history[:, -1, :] = action

        # Update length (capped at context_length)
        self._history_len = torch.clamp(self._history_len + 1, max=self.context_length)

    def _build_transformer_input(self, current_obs: torch.Tensor) -> torch.Tensor:
        """Build transformer input from history and current observation.

        Args:
            current_obs: Current observation (batch, obs_dim)

        Returns:
            Transformer input (batch, seq_len, obs_dim + action_dim)
        """
        batch_size = current_obs.shape[0]
        device = current_obs.device

        # Initialize history if needed
        if self._obs_history is None or self._obs_history.shape[0] != batch_size:
            self._init_history(batch_size, device)

        # Build sequence: [history..., current_obs + zero_action]
        # For the current step, we use zero action as placeholder
        current_action_placeholder = torch.zeros(batch_size, self.num_actions, device=device)

        # Concatenate obs and action for each timestep
        history_tokens = torch.cat([self._obs_history, self._action_history], dim=-1)
        current_token = torch.cat([current_obs, current_action_placeholder], dim=-1).unsqueeze(1)

        # Full sequence: history + current
        sequence = torch.cat([history_tokens, current_token], dim=1)

        return sequence

    def reset(self, dones: torch.Tensor | None = None) -> None:
        """Reset history for done environments.

        Args:
            dones: Boolean tensor indicating which environments are done
        """
        if dones is None or self._obs_history is None:
            return

        # Reset history for done environments
        if dones.any():
            self._obs_history[dones] = 0
            self._action_history[dones] = 0
            self._history_len[dones] = 0

    def forward(self) -> NoReturn:
        raise NotImplementedError

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    def get_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        obs_list = [obs[obs_group] for obs_group in self.obs_groups["policy"]]
        return torch.cat(obs_list, dim=-1)

    def get_critic_obs(self, obs: TensorDict) -> torch.Tensor:
        obs_list = [obs[obs_group] for obs_group in self.obs_groups["critic"]]
        return torch.cat(obs_list, dim=-1)

    def _update_distribution(self, transformer_output: torch.Tensor) -> None:
        """Update action distribution from transformer output."""
        if self.state_dependent_std:
            mean_and_std = self.actor(transformer_output)
            if self.noise_std_type == "scalar":
                mean, std = torch.unbind(mean_and_std, dim=-2)
            else:
                mean, log_std = torch.unbind(mean_and_std, dim=-2)
                std = torch.exp(log_std)
        else:
            mean = self.actor(transformer_output)
            if self.noise_std_type == "scalar":
                std = self.std.expand_as(mean)
            else:
                std = torch.exp(self.log_std).expand_as(mean)
        self.distribution = Normal(mean, std)

    def act(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        """Sample action from policy.

        Args:
            obs: Observation TensorDict

        Returns:
            Sampled action
        """
        # Get current observation
        actor_obs = self.get_actor_obs(obs)
        actor_obs = self.actor_obs_normalizer(actor_obs)

        # Build transformer input with history
        transformer_input = self._build_transformer_input(actor_obs)

        # Forward through transformer
        transformer_output = self.transformer(transformer_input)

        # Get output for current timestep (last position)
        current_output = transformer_output[:, -1, :]

        # Update distribution and sample action
        self._update_distribution(current_output)
        action = self.distribution.sample()

        # Update history with current obs and sampled action
        self._update_history(actor_obs, action)

        return action

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        """Deterministic action for inference.

        Args:
            obs: Observation TensorDict

        Returns:
            Mean action (no sampling)
        """
        actor_obs = self.get_actor_obs(obs)
        actor_obs = self.actor_obs_normalizer(actor_obs)

        transformer_input = self._build_transformer_input(actor_obs)
        transformer_output = self.transformer(transformer_input)
        current_output = transformer_output[:, -1, :]

        if self.state_dependent_std:
            action = self.actor(current_output)[..., 0, :]
        else:
            action = self.actor(current_output)

        # Still update history for subsequent steps
        self._update_history(actor_obs, action)

        return action

    def evaluate(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        """Evaluate value function.

        Note: During training, this is called with batched trajectories.
        We need to process sequences properly.

        Args:
            obs: Observation TensorDict

        Returns:
            Value estimate
        """
        critic_obs = self.get_critic_obs(obs)
        critic_obs = self.critic_obs_normalizer(critic_obs)

        # For single-step evaluation (inference), use history
        if critic_obs.dim() == 2:
            transformer_input = self._build_transformer_input(critic_obs)
            transformer_output = self.transformer(transformer_input)
            current_output = transformer_output[:, -1, :]
        else:
            # Batch mode: obs is (batch, seq_len, obs_dim) - not currently used
            # Fall back to processing without history context
            # This is a simplification - full implementation would track history
            batch_size = critic_obs.shape[0]
            transformer_input = torch.cat([
                critic_obs,
                torch.zeros(batch_size, self.num_actions, device=critic_obs.device)
            ], dim=-1).unsqueeze(1)
            transformer_output = self.transformer(transformer_input)
            current_output = transformer_output[:, -1, :]

        return self.critic(current_output)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def get_hidden_states(self) -> tuple:
        """Return history buffers as 'hidden states' for compatibility."""
        return (self._obs_history, self._action_history)

    def update_normalization(self, obs: TensorDict) -> None:
        if self.actor_obs_normalization:
            actor_obs = self.get_actor_obs(obs)
            self.actor_obs_normalizer.update(actor_obs)
        if self.critic_obs_normalization:
            critic_obs = self.get_critic_obs(obs)
            self.critic_obs_normalizer.update(critic_obs)

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        """Load model state dict."""
        super().load_state_dict(state_dict, strict=strict)
        return True
