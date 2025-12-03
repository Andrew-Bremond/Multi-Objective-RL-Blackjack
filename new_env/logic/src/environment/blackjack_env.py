"""Custom Blackjack environment with doubling and bet sizing"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Optional, Dict, Any


class CustomBlackjackEnv(gym.Env):
    """
    Custom Blackjack environment extending gymnasium's Blackjack-v1.
    
    Features:
    - Two-phase game: betting phase (select bet size) and playing phase
    - Doubling down action
    - Discrete bet sizing (1x, 2x, 3x base bet)
    - Multi-objective reward support
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(
        self,
        bet_multipliers: list = [1, 2, 3],
        base_bet: float = 1.0,
        natural: bool = True,
        sab: bool = True,
        render_mode: Optional[str] = None,
    ):
        """
        Initialize custom Blackjack environment.
        
        Args:
            bet_multipliers: List of bet multipliers (e.g., [1, 2, 3])
            base_bet: Base bet amount
            natural: Whether to give higher reward for natural blackjack
            sab: Whether to use "Dealer uses a Soft 17" rule
            render_mode: Rendering mode
        """
        super().__init__()
        
        self.bet_multipliers = bet_multipliers
        self.base_bet = base_bet
        self.natural = natural
        self.sab = sab
        
        # Create base Blackjack environment
        self.base_env = gym.make("Blackjack-v1", natural=natural, sab=sab, render_mode=render_mode)
        
        # Game phases
        self.PHASE_BETTING = 0
        self.PHASE_PLAYING = 1
        
        # Action spaces
        # Betting phase: select bet multiplier (0, 1, 2 for [1x, 2x, 3x])
        self.betting_action_space = spaces.Discrete(len(bet_multipliers))
        
        # Playing phase: Hit (0), Stand (1), Double (2)
        self.playing_action_space = spaces.Discrete(3)
        
        # Combined action space: (phase, action)
        # For betting: (0, bet_idx)
        # For playing: (1, game_action)
        self.action_space = spaces.MultiDiscrete([2, 3])
        
        # State space: [player_sum, dealer_card, usable_ace, bet_multiplier, phase, can_double]
        # player_sum: 4-31 (28 values)
        # dealer_card: 1-10 (10 values)
        # usable_ace: 0 or 1 (2 values)
        # bet_multiplier: normalized (0-1)
        # phase: 0 or 1 (2 values)
        # can_double: 0 or 1 (2 values)
        self.observation_space = spaces.Box(
            low=np.array([0, 1, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([31, 10, 1, 1, 1, 1], dtype=np.float32),
            dtype=np.float32
        )
        
        # Game state
        self.phase = self.PHASE_BETTING
        self.current_bet_multiplier = None
        self.can_double = False
        self.doubled = False
        
        self.render_mode = render_mode
        
    def _get_obs(self) -> np.ndarray:
        """Get current observation."""
        # Get state from stored values (set during reset and updated during play)
        player_sum = getattr(self, '_player_sum', 0)
        dealer_card = getattr(self, '_dealer_card', 1)
        usable_ace = getattr(self, '_usable_ace', 0)
        
        # If in playing phase, update from base environment
        if self.phase == self.PHASE_PLAYING:
            try:
                # Access unwrapped environment if it exists
                unwrapped = getattr(self.base_env, 'unwrapped', self.base_env)
                if hasattr(unwrapped, '_get_obs'):
                    player_sum, dealer_card, usable_ace = unwrapped._get_obs()
                    self._player_sum = player_sum
                    self._dealer_card = dealer_card
                    self._usable_ace = usable_ace
            except (AttributeError, TypeError):
                # Fallback to stored values
                pass
        
        bet_mult = (self.current_bet_multiplier / max(self.bet_multipliers)) if self.current_bet_multiplier is not None else 0.0
        phase_val = float(self.phase)
        can_double_val = float(self.can_double and not self.doubled)
        
        return np.array([
            float(player_sum),
            float(dealer_card),
            float(usable_ace),
            bet_mult,
            phase_val,
            can_double_val
        ], dtype=np.float32)
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional info."""
        info = {
            "phase": "betting" if self.phase == self.PHASE_BETTING else "playing",
            "bet_multiplier": self.current_bet_multiplier,
            "can_double": self.can_double and not self.doubled,
            "doubled": self.doubled,
        }
        
        # Try to get info from base environment if available
        if self.phase == self.PHASE_PLAYING:
            try:
                # Access unwrapped environment if it exists
                unwrapped = getattr(self.base_env, 'unwrapped', self.base_env)
                if hasattr(unwrapped, '_get_info'):
                    base_info = unwrapped._get_info()
                    info.update(base_info)
            except (AttributeError, TypeError):
                # If we can't get base info, just use our own
                pass
        
        return info
    
    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment."""
        super().reset(seed=seed)
        
        # Reset base environment to get initial state
        obs, info = self.base_env.reset(seed=seed, options=options)
        self._player_sum, self._dealer_card, self._usable_ace = obs
        
        # Reset game state
        self.phase = self.PHASE_BETTING
        self.current_bet_multiplier = None
        self.can_double = False
        self.doubled = False
        
        return self._get_obs(), self._get_info()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Step environment.
        
        Args:
            action: [phase, action_idx]
                - If phase=0 (betting): action_idx is bet multiplier index
                - If phase=1 (playing): action_idx is game action (0=Hit, 1=Stand, 2=Double)
        """
        phase, action_idx = action[0], int(action[1])
        reward = 0.0
        terminated = False
        truncated = False
        info = {}
        
        if self.phase == self.PHASE_BETTING:
            # Betting phase: select bet multiplier
            if action_idx >= len(self.bet_multipliers):
                action_idx = len(self.bet_multipliers) - 1
            
            self.current_bet_multiplier = self.bet_multipliers[action_idx]
            self.phase = self.PHASE_PLAYING
            self.can_double = True  # Can double after initial deal
            
            # State is already available from reset
            # No need to step base environment yet - we're at the start of the game
            
            return self._get_obs(), 0.0, False, False, self._get_info()
        
        else:
            # Playing phase: game actions
            if action_idx == 2 and self.can_double and not self.doubled:
                # Double down
                self.doubled = True
                self.can_double = False
                # Take one card and stand
                obs, _, terminated, truncated, info = self.base_env.step(0)  # Hit
                if not terminated:
                    # Stand after doubling
                    obs, reward, terminated, truncated, info = self.base_env.step(1)  # Stand
            elif action_idx == 0:
                # Hit
                obs, reward, terminated, truncated, info = self.base_env.step(0)
            elif action_idx == 1:
                # Stand
                obs, reward, terminated, truncated, info = self.base_env.step(1)
            else:
                # Invalid action (e.g., double when not available), treat as stand
                obs, reward, terminated, truncated, info = self.base_env.step(1)
            
            # Update state
            if not terminated and not truncated:
                self._player_sum, self._dealer_card, self._usable_ace = obs
            elif terminated or truncated:
                # Game ended, state is final
                self._player_sum, self._dealer_card, self._usable_ace = obs
            
            # Scale reward by bet multiplier
            if terminated or truncated:
                bet_amount = self.base_bet * self.current_bet_multiplier
                reward = reward * bet_amount
                
                # Add info about game outcome
                info['bet_amount'] = bet_amount
                info['raw_reward'] = reward / bet_amount if bet_amount > 0 else 0.0
                info['won'] = reward > 0
                info['lost'] = reward < 0
                info['tie'] = reward == 0
        
        return self._get_obs(), reward, terminated, truncated, info
    
    def render(self):
        """Render environment."""
        if self.render_mode == "human":
            print(f"Phase: {'Betting' if self.phase == self.PHASE_BETTING else 'Playing'}")
            if self.current_bet_multiplier:
                print(f"Bet Multiplier: {self.current_bet_multiplier}x")
            if self.phase == self.PHASE_PLAYING:
                self.base_env.render()
    
    def close(self):
        """Close environment."""
        self.base_env.close()

