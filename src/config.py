from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RewardWeights:
    profit: float = 1.0
    loss_penalty: float = 0.0
    bust_penalty: float = 0.0

    def as_tuple(self):
        return self.profit, self.loss_penalty, self.bust_penalty


@dataclass
class PPOHyperParams:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    learning_rate: float = 3e-4
    update_epochs: int = 4
    minibatch_size: int = 256
    rollout_length: int = 2048
    max_grad_norm: float = 0.5


@dataclass
class TrainSettings:
    total_episodes: int = 10000
    seed: int = 0
    eval_interval: int = 1000
    eval_episodes: int = 200
    deterministic_eval: bool = True


def preset_weights(mode: str) -> RewardWeights:
    mode = mode.lower()
    if mode == "baseline":
        return RewardWeights(profit=1.0, loss_penalty=0.0, bust_penalty=0.0)
    if mode == "risk_averse":
        return RewardWeights(profit=0.6, loss_penalty=0.2, bust_penalty=0.2)
    raise ValueError(f"Unknown mode '{mode}'. Expected 'baseline' or 'risk_averse'.")
