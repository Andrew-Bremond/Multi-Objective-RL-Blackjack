from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RewardWeights:
    # Core objectives (3)
    profit: float = 1.0
    loss_penalty: float = 0.0
    bust_penalty: float = 0.0
    
    # Strategic objectives (5)
    close_call_bonus: float = 0.0  # Reward for getting 19-21
    dealer_weak_bonus: float = 0.0  # Extra reward for winning against weak dealer (2-6)
    conservative_stand_bonus: float = 0.0  # Reward for standing on 12-16 vs dealer 2-6
    aggressive_hit_bonus: float = 0.0  # Reward for successful hits on 12-16
    perfect_play_bonus: float = 0.0  # Bonus for 20-21 final hands
    
    # Advanced objectives (4)
    blackjack_bonus: float = 0.0  # Extra reward for natural 21
    push_penalty: float = 0.0  # Penalty for ties
    early_stand_penalty: float = 0.0  # Penalty for standing on low values (<17)
    dealer_bust_bonus: float = 0.0  # Extra reward when dealer busts
    
    def as_tuple(self):
        return (self.profit, self.loss_penalty, self.bust_penalty,
                self.close_call_bonus, self.dealer_weak_bonus,
                self.conservative_stand_bonus, self.aggressive_hit_bonus,
                self.perfect_play_bonus, self.blackjack_bonus,
                self.push_penalty, self.early_stand_penalty,
                self.dealer_bust_bonus)


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


def preset_weights(mode: str) -> RewardWeights:
    mode = mode.lower()
    if mode == "baseline":
        return RewardWeights(profit=1.0, loss_penalty=0.0, bust_penalty=0.0)
    if mode == "risk_averse":
        return RewardWeights(profit=0.6, loss_penalty=0.2, bust_penalty=0.2)
    if mode == "strategic":
        # New strategic mode with multiple objectives
        return RewardWeights(
            profit=0.7,
            loss_penalty=0.05,
            bust_penalty=0.1,
            close_call_bonus=0.05,  # Encourage getting close to 21
            dealer_weak_bonus=0.05,  # Exploit weak dealer hands
            conservative_stand_bonus=0.02,  # Smart standing
            aggressive_hit_bonus=0.03,  # Reward calculated risks
            perfect_play_bonus=0.05  # Bonus for 20-21
        )
    if mode == "optimal":
        # Optimized for maximum win rate
        return RewardWeights(
            profit=0.8,
            loss_penalty=0.03,
            bust_penalty=0.07,
            close_call_bonus=0.03,
            dealer_weak_bonus=0.03,
            conservative_stand_bonus=0.01,
            aggressive_hit_bonus=0.02,
            perfect_play_bonus=0.01
        )
    raise ValueError(f"Unknown mode '{mode}'. Expected 'baseline', 'risk_averse', 'strategic', or 'optimal'.")
