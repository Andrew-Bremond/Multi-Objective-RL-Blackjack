"""Base agent interface for RL algorithms"""

from abc import ABC, abstractmethod
import torch
from typing import Dict, Any, Optional
import numpy as np


class BaseAgent(ABC):
    """Abstract base class for RL agents."""
    
    def __init__(self, state_dim: int, action_dim: int, device: str = "cpu"):
        """
        Initialize base agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            device: Device to run on
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.training = True
    
    @abstractmethod
    def act(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Select action given state.
        
        Args:
            state: Current state
            deterministic: Whether to select action deterministically
        
        Returns:
            Selected action
        """
        pass
    
    @abstractmethod
    def train_step(self, batch: Any) -> Dict[str, float]:
        """
        Perform one training step.
        
        Args:
            batch: Batch of experiences
        
        Returns:
            Dictionary of training metrics
        """
        pass
    
    @abstractmethod
    def save(self, filepath: str):
        """Save agent to file."""
        pass
    
    @abstractmethod
    def load(self, filepath: str):
        """Load agent from file."""
        pass
    
    def eval(self):
        """Set agent to evaluation mode."""
        self.training = False
    
    def train(self):
        """Set agent to training mode."""
        self.training = True

