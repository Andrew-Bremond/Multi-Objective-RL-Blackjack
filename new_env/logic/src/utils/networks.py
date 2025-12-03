"""Neural network architectures for RL algorithms"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DQNNetwork(nn.Module):
    """Standard DQN network architecture."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list = [128, 128]):
        """
        Initialize DQN network.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dims: List of hidden layer dimensions
        """
        super(DQNNetwork, self).__init__()
        
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(state)


class DuelingDQNNetwork(nn.Module):
    """Dueling DQN network with separate value and advantage streams."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list = [128, 128]):
        """
        Initialize Dueling DQN network.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dims: List of hidden layer dimensions
        """
        super(DuelingDQNNetwork, self).__init__()
        
        # Shared feature extractor
        shared_layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims[:-1]:
            shared_layers.append(nn.Linear(input_dim, hidden_dim))
            shared_layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        self.shared = nn.Sequential(*shared_layers)
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], action_dim)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        shared_features = self.shared(state)
        
        value = self.value_stream(shared_features)
        advantage = self.advantage_stream(shared_features)
        
        # Combine value and advantage
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = value + (advantage - advantage.mean(dim=-1, keepdim=True))
        
        return q_values


class ActorCritic(nn.Module):
    """Actor-Critic network for PPO."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list = [128, 128],
        continuous: bool = False,
    ):
        """
        Initialize Actor-Critic network.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dims: List of hidden layer dimensions
            continuous: Whether actions are continuous (False for discrete)
        """
        super(ActorCritic, self).__init__()
        
        self.continuous = continuous
        
        # Shared feature extractor
        shared_layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims[:-1]:
            shared_layers.append(nn.Linear(input_dim, hidden_dim))
            shared_layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        self.shared = nn.Sequential(*shared_layers)
        
        # Actor (policy) head
        if continuous:
            # For continuous actions, output mean and std
            self.actor_mean = nn.Sequential(
                nn.Linear(input_dim, hidden_dims[-1]),
                nn.ReLU(),
                nn.Linear(hidden_dims[-1], action_dim),
                nn.Tanh()
            )
            self.actor_std = nn.Sequential(
                nn.Linear(input_dim, hidden_dims[-1]),
                nn.ReLU(),
                nn.Linear(hidden_dims[-1], action_dim),
                nn.Softplus()
            )
        else:
            # For discrete actions, output action probabilities
            self.actor = nn.Sequential(
                nn.Linear(input_dim, hidden_dims[-1]),
                nn.ReLU(),
                nn.Linear(hidden_dims[-1], action_dim),
                nn.Softmax(dim=-1)
            )
        
        # Critic (value) head
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], 1)
        )
    
    def forward(self, state: torch.Tensor):
        """Forward pass."""
        shared_features = self.shared(state)
        
        if self.continuous:
            mean = self.actor_mean(shared_features)
            std = self.actor_std(shared_features)
            value = self.critic(shared_features)
            return mean, std, value
        else:
            action_probs = self.actor(shared_features)
            value = self.critic(shared_features)
            return action_probs, value
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False):
        """
        Sample action from policy.
        
        Args:
            state: Current state
            deterministic: If True, return deterministic action
        
        Returns:
            action, log_prob, value
        """
        if self.continuous:
            mean, std, value = self.forward(state)
            dist = torch.distributions.Normal(mean, std)
            
            if deterministic:
                action = mean
            else:
                action = dist.sample()
            
            log_prob = dist.log_prob(action).sum(dim=-1)
            return action, log_prob, value
        else:
            action_probs, value = self.forward(state)
            dist = torch.distributions.Categorical(action_probs)
            
            if deterministic:
                action = torch.argmax(action_probs, dim=-1)
            else:
                action = dist.sample()
            
            log_prob = dist.log_prob(action)
            return action, log_prob, value
    
    def evaluate_actions(self, state: torch.Tensor, action: torch.Tensor):
        """
        Evaluate actions and return log probs, entropy, and value.
        
        Args:
            state: Current state
            action: Actions to evaluate
        
        Returns:
            log_prob, entropy, value
        """
        if self.continuous:
            mean, std, value = self.forward(state)
            dist = torch.distributions.Normal(mean, std)
            log_prob = dist.log_prob(action).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
        else:
            action_probs, value = self.forward(state)
            dist = torch.distributions.Categorical(action_probs)
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
        
        return log_prob, entropy, value

