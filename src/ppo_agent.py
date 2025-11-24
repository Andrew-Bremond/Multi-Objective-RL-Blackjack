from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: Tuple[int, int] = (128, 128)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
        )
        self.actor = nn.Linear(hidden_sizes[1], action_dim)
        self.critic = nn.Linear(hidden_sizes[1], 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.net(x)
        return self.actor(x), self.critic(x)


@dataclass
class Rollout:
    observations: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    values: torch.Tensor
    last_value: float


class PPOAgent:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_coef: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        device: str | torch.device = "cpu",
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm

        self.model = ActorCritic(obs_dim, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> Tuple[int, float, float]:
        obs_tensor = torch.from_numpy(obs).float().to(self.device).unsqueeze(0)
        logits, value = self.model(obs_tensor)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return int(action.item()), float(log_prob.item()), float(value.squeeze().item())

    def _evaluate_actions(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, values = self.model(observations)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy, values.squeeze(-1)

    @torch.no_grad()
    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> int:
        obs_tensor = torch.from_numpy(obs).float().to(self.device).unsqueeze(0)
        logits, _ = self.model(obs_tensor)
        dist = Categorical(logits=logits)
        if deterministic:
            action = torch.argmax(dist.probs, dim=-1)
        else:
            action = dist.sample()
        return int(action.item())

    def compute_gae(
        self, rewards: torch.Tensor, dones: torch.Tensor, values: torch.Tensor, last_value: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generalized Advantage Estimation with bootstrapped final value."""
        values_ext = torch.cat([values, torch.tensor([last_value], device=self.device, dtype=values.dtype)])
        advantages = torch.zeros_like(rewards, device=self.device)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            next_value = values_ext[t + 1]
            next_non_terminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages[t] = gae
        returns = advantages + values
        return advantages, returns

    def update(
        self,
        rollout: Rollout,
        update_epochs: int,
        minibatch_size: int,
    ) -> Dict[str, float]:
        obs = rollout.observations.to(self.device)
        actions = rollout.actions.to(self.device)
        old_log_probs = rollout.log_probs.to(self.device)
        rewards = rollout.rewards.to(self.device)
        dones = rollout.dones.to(self.device)
        values = rollout.values.to(self.device)

        with torch.no_grad():
            advantages, returns = self.compute_gae(rewards, dones, values, rollout.last_value)
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        num_updates = 0

        batch_size = len(rewards)
        indices = np.arange(batch_size)

        for _ in range(update_epochs):
            np.random.shuffle(indices)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_idx = indices[start:end]
                mb_idx_tensor = torch.as_tensor(mb_idx, device=self.device, dtype=torch.long)

                mb_obs = obs[mb_idx_tensor]
                mb_actions = actions[mb_idx_tensor]
                mb_old_log_probs = old_log_probs[mb_idx_tensor]
                mb_adv = advantages[mb_idx_tensor]
                mb_returns = returns[mb_idx_tensor]

                new_log_probs, entropy, new_values = self._evaluate_actions(mb_obs, mb_actions)
                log_ratio = new_log_probs - mb_old_log_probs
                ratio = torch.exp(log_ratio)

                unclipped = ratio * mb_adv
                clipped = torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef) * mb_adv
                policy_loss = -torch.min(unclipped, clipped).mean()

                value_loss = F.mse_loss(new_values, mb_returns)
                entropy_loss = entropy.mean()

                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy_loss.item()
                num_updates += 1

        return {
            "policy_loss": total_policy_loss / max(1, num_updates),
            "value_loss": total_value_loss / max(1, num_updates),
            "entropy": total_entropy / max(1, num_updates),
        }
