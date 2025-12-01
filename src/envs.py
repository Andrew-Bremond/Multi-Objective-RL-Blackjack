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
        self.prev_player_sum = None
        self.prev_action = None
        self.initial_dealer_card = None

    def step(self, action: int):
        obs, base_reward, terminated, truncated, info = self.env.step(action)
        info = info or {}

        player_sum, dealer_card, usable_ace = obs
        is_bust = player_sum > 21
        is_win = base_reward > 0
        is_loss = base_reward < 0
        done = terminated or truncated
        
        # Track additional strategic information
        is_close_call = 19 <= player_sum <= 21 and not is_bust
        is_perfect = 20 <= player_sum <= 21 and not is_bust
        dealer_is_weak = self.initial_dealer_card in [2, 3, 4, 5, 6] if self.initial_dealer_card else False
        
        # Conservative stand: standing on 12-16 against weak dealer
        conservative_stand = (
            action == 0 and  # Stand action
            12 <= self.prev_player_sum <= 16 and
            dealer_is_weak and
            not is_bust and
            done
        ) if self.prev_player_sum else False
        
        # Aggressive hit: successful hit on risky hand
        aggressive_hit_success = (
            self.prev_action == 1 and  # Hit action
            self.prev_player_sum and
            12 <= self.prev_player_sum <= 16 and
            not is_bust and
            player_sum > self.prev_player_sum
        ) if self.prev_action is not None else False

        # Calculate multi-objective reward
        scalar_reward = (
            self.weights.profit * float(base_reward)
            - self.weights.loss_penalty * float(is_loss)
            - self.weights.bust_penalty * float(is_bust)
            + self.weights.close_call_bonus * float(is_close_call and done and is_win)
            + self.weights.dealer_weak_bonus * float(dealer_is_weak and is_win)
            + self.weights.conservative_stand_bonus * float(conservative_stand and not is_loss)
            + self.weights.aggressive_hit_bonus * float(aggressive_hit_success)
            + self.weights.perfect_play_bonus * float(is_perfect and done and is_win)
        )

        info.update(
            {
                "base_reward": float(base_reward),
                "scalar_reward": float(scalar_reward),
                "is_bust": bool(is_bust),
                "is_win": bool(is_win),
                "is_loss": bool(is_loss),
                "is_close_call": bool(is_close_call),
                "is_perfect": bool(is_perfect),
                "dealer_weak": bool(dealer_is_weak),
                "conservative_stand": bool(conservative_stand),
                "aggressive_hit": bool(aggressive_hit_success),
            }
        )
        
        # Update tracking variables
        self.prev_player_sum = player_sum if not done else None
        self.prev_action = action if not done else None

        return obs, scalar_reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Store initial dealer card for strategy tracking
        _, dealer_card, _ = obs
        self.initial_dealer_card = dealer_card
        self.prev_player_sum = None
        self.prev_action = None
        return obs, info


def make_env(weights: RewardWeights, seed: int | None = None) -> MultiObjectiveBlackjack:
    env = gym.make("Blackjack-v1", natural=False, sab=False)
    env.reset(seed=seed)
    return MultiObjectiveBlackjack(env, weights)
