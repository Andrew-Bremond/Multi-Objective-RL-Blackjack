"""Multi-objective reward framework for Blackjack"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import deque


class MultiObjectiveReward:
    """
    Multi-objective reward framework for Blackjack.
    
    Objectives:
    1. Expected Return - Maximize average profit per game
    2. Risk-Adjusted Return (Sharpe Ratio) - Balance return with volatility
    3. Win Rate - Maximize percentage of winning games
    """
    
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        rolling_window: int = 100,
        risk_free_rate: float = 0.0,
    ):
        """
        Initialize multi-objective reward framework.
        
        Args:
            weights: Dictionary with weights for each objective
                Keys: 'expected_return', 'sharpe_ratio', 'win_rate'
            rolling_window: Window size for computing rolling statistics
            risk_free_rate: Risk-free rate for Sharpe ratio calculation
        """
        if weights is None:
            weights = {
                'expected_return': 0.5,
                'sharpe_ratio': 0.3,
                'win_rate': 0.2,
            }
        
        self.weights = weights
        self.rolling_window = rolling_window
        self.risk_free_rate = risk_free_rate
        
        # Track returns for Sharpe ratio calculation
        self.returns_history = deque(maxlen=rolling_window)
        
        # Track wins for win rate calculation
        self.wins_history = deque(maxlen=rolling_window)
        self.games_history = deque(maxlen=rolling_window)
        
        # Episode-level tracking
        self.episode_returns = []
        self.episode_wins = []
        self.episode_count = 0
    
    def compute_reward(
        self,
        raw_reward: float,
        game_ended: bool = False,
        won: bool = False,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute multi-objective reward.
        
        Args:
            raw_reward: Raw reward from environment (scaled by bet)
            game_ended: Whether the game has ended
            won: Whether the game was won (for win rate calculation)
        
        Returns:
            Tuple of (combined_reward, objective_values)
        """
        objective_values = {}
        
        # Objective 1: Expected Return
        expected_return = raw_reward
        objective_values['expected_return'] = expected_return
        
        # Objective 2: Sharpe Ratio (computed over rolling window)
        if game_ended:
            self.returns_history.append(raw_reward)
            self.games_history.append(1)
            if won:
                self.wins_history.append(1)
            else:
                self.wins_history.append(0)
        
        sharpe_ratio = self._compute_sharpe_ratio()
        objective_values['sharpe_ratio'] = sharpe_ratio
        
        # Objective 3: Win Rate
        win_rate = self._compute_win_rate()
        objective_values['win_rate'] = win_rate
        
        # Combine objectives with weights
        combined_reward = (
            self.weights['expected_return'] * expected_return +
            self.weights['sharpe_ratio'] * sharpe_ratio +
            self.weights['win_rate'] * win_rate
        )
        
        return combined_reward, objective_values
    
    def _compute_sharpe_ratio(self) -> float:
        """Compute Sharpe ratio from returns history."""
        if len(self.returns_history) < 2:
            return 0.0
        
        returns_array = np.array(self.returns_history)
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array)
        
        if std_return == 0:
            return 0.0
        
        # Annualized Sharpe ratio (assuming daily returns, adjust as needed)
        sharpe = (mean_return - self.risk_free_rate) / std_return
        
        # Normalize to reasonable range for reward shaping
        # Clip to [-1, 1] range
        sharpe_normalized = np.clip(sharpe / 10.0, -1.0, 1.0)
        
        return sharpe_normalized
    
    def _compute_win_rate(self) -> float:
        """Compute win rate from history."""
        if len(self.wins_history) == 0:
            return 0.0
        
        win_rate = np.mean(self.wins_history)
        # Normalize: convert from [0, 1] to [-1, 1] for better reward shaping
        win_rate_normalized = (win_rate - 0.5) * 2.0
        
        return win_rate_normalized
    
    def update_episode(self, episode_return: float, episode_wins: int, episode_games: int):
        """
        Update episode-level statistics.
        
        Args:
            episode_return: Total return for the episode
            episode_wins: Number of wins in the episode
            episode_games: Number of games in the episode
        """
        self.episode_returns.append(episode_return)
        if episode_games > 0:
            self.episode_wins.append(episode_wins / episode_games)
        else:
            self.episode_wins.append(0.0)
        self.episode_count += 1
    
    def get_metrics(self) -> Dict[str, float]:
        """Get current metrics for all objectives."""
        return {
            'expected_return': np.mean(self.episode_returns[-self.rolling_window:]) if self.episode_returns else 0.0,
            'sharpe_ratio': self._compute_sharpe_ratio(),
            'win_rate': np.mean(self.wins_history) if self.wins_history else 0.0,
            'episode_count': self.episode_count,
        }
    
    def reset(self):
        """Reset tracking (keep history for rolling window)."""
        # Keep history for rolling statistics
        # Only reset episode-level tracking if needed
        pass
    
    def set_weights(self, weights: Dict[str, float]):
        """Update objective weights."""
        self.weights = weights

