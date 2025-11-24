from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np

from .config import RewardWeights


def normalize_obs(obs: Tuple[int, int, bool]) -> np.ndarray:
    """Scale observation into a float vector in [0, 1]."""
    player_sum, dealer_card, usable_ace = obs
    return np.array(
        [player_sum / 32.0, dealer_card / 11.0, float(usable_ace)],
        dtype=np.float32,
    )


@dataclass
class Outcome:
    reward: float
    scalar_reward: float
    is_win: bool
    is_loss: bool
    is_bust: bool


class MultiObjectiveBlackjack(gym.Wrapper):
    """Wraps Gymnasium Blackjack to add scalarized multi-objective rewards."""

    def __init__(self, env: gym.Env, weights: RewardWeights):
        super().__init__(env)
        self.weights = weights

    def step(self, action: int):
        obs, base_reward, terminated, truncated, info = self.env.step(action)
        info = info or {}

        player_sum, _, _ = obs
        is_bust = player_sum > 21
        is_win = base_reward > 0
        is_loss = base_reward < 0

        scalar_reward = (
            self.weights.profit * float(base_reward)
            - self.weights.loss_penalty * float(is_loss)
            - self.weights.bust_penalty * float(is_bust)
        )

        info.update(
            {
                "base_reward": float(base_reward),
                "scalar_reward": float(scalar_reward),
                "is_bust": bool(is_bust),
                "is_win": bool(is_win),
                "is_loss": bool(is_loss),
            }
        )

        return obs, scalar_reward, terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


def make_env(weights: RewardWeights, seed: int | None = None) -> MultiObjectiveBlackjack:
    env = gym.make("Blackjack-v1", natural=False, sab=False)
    env.reset(seed=seed)
    return MultiObjectiveBlackjack(env, weights)
