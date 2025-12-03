"""Double DQN (DDQN) implementation"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, Optional
import random

from .base_agent import BaseAgent
from .dqn import DQN
from ..utils.networks import DQNNetwork
from ..utils.replay_buffer import ReplayBuffer


class DDQN(DQN):
    """Double DQN agent - uses main network for action selection, target network for evaluation."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int = 6,
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
        """Initialize Double DQN agent (same as DQN)."""
        super().__init__(
            state_dim, action_dim, lr, gamma, epsilon_start, epsilon_end,
            epsilon_decay, batch_size, buffer_size, target_update_freq,
            device, hidden_dims
        )
    
    def train_step(self, batch: Optional[Any] = None) -> Dict[str, float]:
        """Perform one training step with Double DQN update."""
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
        
        # Double DQN: use main network to select actions, target network to evaluate
        with torch.no_grad():
            next_q_values_main = self.q_network(next_states)
            next_actions = next_q_values_main.argmax(1)
            next_q_values_target = self.target_network(next_states)
            next_q_value = next_q_values_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
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

