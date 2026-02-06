"""PPO with KL Distillation Algorithm.

Implements the training approach from Radosavovic et al. (2024):
L_total = L_PPO + λ * D_KL(π_student || π_teacher)

Where λ is annealed from λ_init to 0 at training midpoint.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic
from rsl_rl.storage import RolloutStorage


class DistillationPPO(PPO):
    """PPO algorithm with KL distillation from a teacher policy.

    This extends PPO to add a KL divergence loss term that encourages
    the student to match the teacher's action distribution.

    Loss = L_PPO + λ * D_KL(π_student || π_teacher)

    Where λ is annealed according to the paper's schedule.
    """

    def __init__(
        self,
        actor_critic: ActorCritic,
        teacher_actor_critic: ActorCritic,
        # PPO params
        num_learning_epochs: int = 1,
        num_mini_batches: int = 1,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.0,
        learning_rate: float = 1e-3,
        max_grad_norm: float = 1.0,
        use_clipped_value_loss: bool = True,
        schedule: str = "fixed",
        desired_kl: float = 0.01,
        device: str = "cpu",
        # Distillation params
        kl_coef_init: float = 1.0,
        kl_anneal_midpoint: float = 0.5,
    ):
        """Initialize PPO with distillation.

        Args:
            actor_critic: Student actor-critic network
            teacher_actor_critic: Teacher actor-critic network (frozen)
            kl_coef_init: Initial KL divergence coefficient (λ_init)
            kl_anneal_midpoint: Fraction of training where λ reaches 0
            ... other PPO args
        """
        super().__init__(
            actor_critic=actor_critic,
            num_learning_epochs=num_learning_epochs,
            num_mini_batches=num_mini_batches,
            clip_param=clip_param,
            gamma=gamma,
            lam=lam,
            value_loss_coef=value_loss_coef,
            entropy_coef=entropy_coef,
            learning_rate=learning_rate,
            max_grad_norm=max_grad_norm,
            use_clipped_value_loss=use_clipped_value_loss,
            schedule=schedule,
            desired_kl=desired_kl,
            device=device,
        )

        # Teacher network (frozen)
        self.teacher_actor_critic = teacher_actor_critic
        self.teacher_actor_critic.eval()
        for param in self.teacher_actor_critic.parameters():
            param.requires_grad = False

        # Distillation parameters
        self.kl_coef_init = kl_coef_init
        self.kl_anneal_midpoint = kl_anneal_midpoint
        self.current_kl_coef = kl_coef_init

        # Tracking
        self.mean_kl_loss = 0.0

    def update_kl_coef(self, current_iteration: int, max_iterations: int):
        """Update KL coefficient based on training progress.

        Linear anneal from kl_coef_init to 0 at midpoint, then stays at 0.
        """
        progress = current_iteration / max_iterations
        anneal_progress = min(1.0, progress / self.kl_anneal_midpoint)
        self.current_kl_coef = self.kl_coef_init * (1.0 - anneal_progress)

    def compute_kl_divergence(
        self,
        student_obs: torch.Tensor,
        teacher_obs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute KL divergence between student and teacher action distributions.

        D_KL(π_student || π_teacher) = E[log(π_student) - log(π_teacher)]

        For Gaussian policies:
        D_KL = log(σ_t/σ_s) + (σ_s² + (μ_s - μ_t)²) / (2σ_t²) - 0.5
        """
        # Get student distribution
        self.actor_critic.act(student_obs)
        student_mean = self.actor_critic.action_mean
        student_std = self.actor_critic.action_std

        # Get teacher distribution (no gradients)
        with torch.no_grad():
            self.teacher_actor_critic.act(teacher_obs)
            teacher_mean = self.teacher_actor_critic.action_mean
            teacher_std = self.teacher_actor_critic.action_std

        # Create distributions
        student_dist = Normal(student_mean, student_std)
        teacher_dist = Normal(teacher_mean, teacher_std)

        # Compute KL divergence
        kl_div = torch.distributions.kl_divergence(student_dist, teacher_dist)

        # Sum over action dimensions, mean over batch
        kl_loss = kl_div.sum(dim=-1).mean()

        return kl_loss

    def update(self, storage: RolloutStorage):
        """Update policy with PPO + KL distillation loss.

        Extends parent PPO.update() to add KL divergence term.
        """
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_kl_loss = 0

        if self.actor_critic.is_recurrent:
            generator = storage.recurrent_mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )
        else:
            generator = storage.mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )

        for (
            obs_batch,
            critic_obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
        ) in generator:

            # === Standard PPO losses ===
            self.actor_critic.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch)
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(
                critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch
            )
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            # KL divergence for adaptive learning rate
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss (policy loss)
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            # === KL Distillation Loss ===
            if self.current_kl_coef > 0:
                # For teacher, we use the same observations (assuming teacher sees same obs space)
                # In practice, teacher may have privileged state - but for action distillation,
                # we compare action distributions on the STUDENT's observation
                kl_loss = self.compute_kl_divergence(obs_batch, obs_batch)
            else:
                kl_loss = torch.tensor(0.0, device=self.device)

            # === Total Loss ===
            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch.mean()
                + self.current_kl_coef * kl_loss
            )

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_kl_loss += kl_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_kl_loss /= num_updates

        self.mean_kl_loss = mean_kl_loss

        return mean_value_loss, mean_surrogate_loss
