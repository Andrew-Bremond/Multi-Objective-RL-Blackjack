"""Deep Q-Network (DQN) implementation"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, Optional
import random

from .base_agent import BaseAgent
from ..utils.networks import DQNNetwork
from ..utils.replay_buffer import ReplayBuffer


class DQN(BaseAgent):
    """Deep Q-Network agent with experience replay and target network."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int = 6,  # 2 phases * 3 actions (betting: 3 bet sizes, playing: 3 actions)
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        batch_size: int = 64,
        buffer_size: int = 10000,
        target_update_freq: int = 100,
        device: str = "cpu",
        hidden_dims: list = [128, 128],
    ):
        """
        Initialize DQN agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Total action dimension (flattened from phase + action)
            lr: Learning rate
            gamma: Discount factor
            epsilon_start: Starting epsilon for epsilon-greedy
            epsilon_end: Final epsilon
            epsilon_decay: Epsilon decay rate
            batch_size: Batch size for training
            buffer_size: Size of replay buffer
            target_update_freq: Frequency of target network updates
            device: Device to run on
            hidden_dims: Hidden layer dimensions
        """
        super().__init__(state_dim, action_dim, device)
        
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.steps = 0
        
        # Q-network and target network
        self.q_network = DQNNetwork(state_dim, action_dim, hidden_dims).to(device)
        self.target_network = DQNNetwork(state_dim, action_dim, hidden_dims).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size, state_dim, action_dim=2, device=device)
        
        # Loss function
        self.criterion = nn.MSELoss()
    
    def _flatten_action(self, phase: int, action: int) -> int:
        """Flatten (phase, action) to single action index."""
        # Phase 0 (betting): actions 0-2 (bet sizes)
        # Phase 1 (playing): actions 3-5 (hit, stand, double)
        if phase == 0:
            return action  # 0, 1, 2
        else:
            return 3 + action  # 3, 4, 5
    
    def _unflatten_action(self, action_idx: int) -> np.ndarray:
        """Unflatten action index to (phase, action)."""
        if action_idx < 3:
            return np.array([0, action_idx])  # Betting phase
        else:
            return np.array([1, action_idx - 3])  # Playing phase
    
    def act(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select action using epsilon-greedy policy."""
        if not deterministic and random.random() < self.epsilon:
            # Random action
            phase = random.randint(0, 1)
            if phase == 0:
                action = random.randint(0, 2)  # Betting: 3 bet sizes
            else:
                action = random.randint(0, 2)  # Playing: hit, stand, double
            return np.array([phase, action])
        
        # Greedy action
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            action_idx = q_values.argmax().item()
        
        return self._unflatten_action(action_idx)
    
    def train_step(self, batch: Optional[Any] = None) -> Dict[str, float]:
        """Perform one training step."""
        if len(self.replay_buffer) < self.batch_size:
            return {"loss": 0.0}
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Flatten actions for Q-network
        action_indices = []
        for action in actions:
            phase, act = action[0].item(), action[1].item()
            action_idx = self._flatten_action(phase, act)
            action_indices.append(action_idx)
        action_indices = torch.LongTensor(action_indices).to(self.device)
        
        # Current Q-values
        q_values = self.q_network(states)
        q_value = q_values.gather(1, action_indices.unsqueeze(1)).squeeze(1)
        
        # Next Q-values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            next_q_value = next_q_values.max(1)[0]
            target_q_value = rewards + (1 - dones.float()) * self.gamma * next_q_value
        
        # Compute loss
        loss = self.criterion(q_value, target_q_value)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        # Update target network
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        return {"loss": loss.item(), "epsilon": self.epsilon}
    
    def update_replay_buffer(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Add experience to replay buffer."""
        self.replay_buffer.add(state, action, reward, next_state, done)
    
    def save(self, filepath: str):
        """Save agent."""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
        }, filepath)
    
    def load(self, filepath: str):
        """Load agent."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon_start)
        self.steps = checkpoint.get('steps', 0)

