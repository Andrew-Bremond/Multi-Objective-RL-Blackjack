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

        player_sum, dealer_card, usable_ace = obs
        is_bust = player_sum > 21
        is_win = base_reward > 0
        is_loss = base_reward < 0
        done = terminated or truncated
        
        # Track close call: finishing with 19-21 and winning
        is_close_call = 19 <= player_sum <= 21 and not is_bust

        # Calculate multi-objective reward
        scalar_reward = (
            self.weights.profit * float(base_reward)
            - self.weights.loss_penalty * float(is_loss)
            - self.weights.bust_penalty * float(is_bust)
            + self.weights.close_call_bonus * float(is_close_call and done and is_win)
        )

        info.update(
            {
                "base_reward": float(base_reward),
                "scalar_reward": float(scalar_reward),
                "is_bust": bool(is_bust),
                "is_win": bool(is_win),
                "is_loss": bool(is_loss),
                "is_close_call": bool(is_close_call),
            }
        )

        return obs, scalar_reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, info


def make_env(weights: RewardWeights, seed: int | None = None) -> MultiObjectiveBlackjack:
    env = gym.make("Blackjack-v1", natural=False, sab=False)
    env.reset(seed=seed)
    return MultiObjectiveBlackjack(env, weights)
