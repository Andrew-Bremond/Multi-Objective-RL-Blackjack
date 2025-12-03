from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RewardWeights:
    # Core objectives (4)
    profit: float = 1.0
    loss_penalty: float = 0.0
    bust_penalty: float = 0.0
    close_call_bonus: float = 0.0  # Reward for getting 19-21
    
    def as_tuple(self):
        return (self.profit, self.loss_penalty, self.bust_penalty, self.close_call_bonus)


@dataclass
class PPOHyperParams:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    entropy_coef: float = 0.05  # Increased for more exploration
    value_coef: float = 0.5
    learning_rate: float = 5e-4  # Slightly higher for faster learning
    update_epochs: int = 10  # More epochs per update for better learning
    minibatch_size: int = 256
    rollout_length: int = 2048
    max_grad_norm: float = 0.5


@dataclass
class TrainSettings:
    total_episodes: int = 50000  # Much more training for convergence
    seed: int = 0
    eval_interval: int = 1000
    eval_episodes: int = 1000  # More evaluation episodes for accurate metrics
    deterministic_eval: bool = True


def load_weights_from_checkpoint(weights_dict: dict) -> RewardWeights:
    """Load RewardWeights from a checkpoint dictionary, filtering out obsolete fields.
    
    This function ensures backward compatibility with checkpoints saved with the old
    RewardWeights structure that contained additional objectives.
    
    Args:
        weights_dict: Dictionary containing weight values (may include obsolete fields)
        
    Returns:
        RewardWeights instance with only the current valid fields
    """
    # Only extract the fields that exist in the current RewardWeights structure
    valid_fields = {
        "profit": weights_dict.get("profit", 1.0),
        "loss_penalty": weights_dict.get("loss_penalty", 0.0),
        "bust_penalty": weights_dict.get("bust_penalty", 0.0),
        "close_call_bonus": weights_dict.get("close_call_bonus", 0.0),
    }
    return RewardWeights(**valid_fields)


def preset_weights(mode: str) -> RewardWeights:
    mode = mode.lower()
    if mode == "baseline":
        return RewardWeights(profit=1.0, loss_penalty=0.0, bust_penalty=0.0, close_call_bonus=0.0)
    if mode == "risk_averse":
        return RewardWeights(profit=0.6, loss_penalty=0.2, bust_penalty=0.2, close_call_bonus=0.0)
    if mode == "strategic":
        return RewardWeights(
            profit=0.7,
            loss_penalty=0.05,
            bust_penalty=0.1,
            close_call_bonus=0.05  # Encourage getting close to 21
        )
    if mode == "optimal":
        return RewardWeights(
            profit=0.8,
            loss_penalty=0.03,
            bust_penalty=0.07,
            close_call_bonus=0.03
        )
    raise ValueError(f"Unknown mode '{mode}'. Expected 'baseline', 'risk_averse', 'strategic', or 'optimal'.")
