"""Simulate bankroll over multiple games using the custom Blackjack environment"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import yaml
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.environment import CustomBlackjackEnv
from src.algorithms import DQN, DDQN, DuelingDDQN, PPO
import torch


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_agent(algorithm: str, state_dim: int, action_dim: int, config: dict, device: str):
    """Create agent based on algorithm name."""
    algo_config = config['algorithms'][algorithm]
    
    if algorithm == 'dqn':
        return DQN(
            state_dim=state_dim,
            action_dim=action_dim,
            lr=algo_config['lr'],
            gamma=algo_config['gamma'],
            epsilon_start=algo_config['epsilon_start'],
            epsilon_end=algo_config['epsilon_end'],
            epsilon_decay=algo_config['epsilon_decay'],
            batch_size=algo_config['batch_size'],
            buffer_size=algo_config['buffer_size'],
            target_update_freq=algo_config['target_update_freq'],
            device=device,
            hidden_dims=algo_config['hidden_dims'],
        )
    elif algorithm == 'ddqn':
        return DDQN(
            state_dim=state_dim,
            action_dim=action_dim,
            lr=algo_config['lr'],
            gamma=algo_config['gamma'],
            epsilon_start=algo_config['epsilon_start'],
            epsilon_end=algo_config['epsilon_end'],
            epsilon_decay=algo_config['epsilon_decay'],
            batch_size=algo_config['batch_size'],
            buffer_size=algo_config['buffer_size'],
            target_update_freq=algo_config['target_update_freq'],
            device=device,
            hidden_dims=algo_config['hidden_dims'],
        )
    elif algorithm == 'dueling_ddqn':
        return DuelingDDQN(
            state_dim=state_dim,
            action_dim=action_dim,
            lr=algo_config['lr'],
            gamma=algo_config['gamma'],
            epsilon_start=algo_config['epsilon_start'],
            epsilon_end=algo_config['epsilon_end'],
            epsilon_decay=algo_config['epsilon_decay'],
            batch_size=algo_config['batch_size'],
            buffer_size=algo_config['buffer_size'],
            target_update_freq=algo_config['target_update_freq'],
            device=device,
            hidden_dims=algo_config['hidden_dims'],
        )
    elif algorithm == 'ppo':
        return PPO(
            state_dim=state_dim,
            action_dim=action_dim,
            lr=algo_config['lr'],
            gamma=algo_config['gamma'],
            gae_lambda=algo_config['gae_lambda'],
            clip_epsilon=algo_config['clip_epsilon'],
            value_coef=algo_config['value_coef'],
            entropy_coef=algo_config['entropy_coef'],
            max_grad_norm=algo_config['max_grad_norm'],
            ppo_epochs=algo_config['ppo_epochs'],
            batch_size=algo_config['batch_size'],
            device=device,
            hidden_dims=algo_config['hidden_dims'],
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def simulate_bankroll(
    initial_bankroll: float = 100.0,
    num_iterations: int = 100,
    bet_strategy: str = "fixed",
    bet_multiplier: int = 1,
    config_path: str = "configs/config.yaml",
    seed: int = None,
    model_path: str = None,
    algorithm: str = None,
):
    """
    Simulate bankroll over multiple games.
    
    Args:
        initial_bankroll: Starting amount of money
        num_iterations: Number of games to play
        bet_strategy: "fixed" (always bet same), "random" (random bet size), "proportional" (bet % of bankroll)
        bet_multiplier: For fixed strategy, which multiplier to use (1, 2, or 3)
        config_path: Path to config file
        seed: Random seed for reproducibility
    """
    # Load config
    config = load_config(config_path)
    
    # Set device
    device = config['training']['device']
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    
    # Create environment
    env = CustomBlackjackEnv(
        bet_multipliers=config['environment']['bet_multipliers'],
        base_bet=config['environment']['base_bet'],
        natural=config['environment']['natural'],
        sab=config['environment']['sab'],
    )
    
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        env.reset(seed=seed)
    
    # Load agent if model path provided
    agent = None
    use_agent = model_path is not None and algorithm is not None
    if use_agent:
        state_dim = env.observation_space.shape[0]
        action_dim = 6  # Flattened action space
        agent = create_agent(algorithm, state_dim, action_dim, config, device)
        agent.load(model_path)
        agent.eval()
        print(f"Loaded {algorithm.upper()} model from {model_path}")
    
    # Initialize tracking
    bankroll_history = [initial_bankroll]
    current_bankroll = initial_bankroll
    game_results = []
    
    bet_multipliers = config['environment']['bet_multipliers']
    base_bet = config['environment']['base_bet']
    
    print(f"Starting simulation with ${initial_bankroll:.2f}")
    print(f"Playing {num_iterations} games...")
    if use_agent:
        print(f"Using trained {algorithm.upper()} agent for decisions")
    else:
        print(f"Bet strategy: {bet_strategy}")
        if bet_strategy == "fixed":
            print(f"Bet multiplier: {bet_multiplier}x (betting ${base_bet * bet_multiplier:.2f} per game)")
    print("-" * 50)
    
    for iteration in range(num_iterations):
        # Check if we can afford to play
        if current_bankroll <= 0:
            print(f"\nBankrupt at iteration {iteration}! Stopping simulation.")
            break
        
        # Reset environment for new game
        state, info = env.reset()
        
        # Phase 1: Betting
        if use_agent:
            # Use agent to select bet size
            agent_result = agent.act(state, deterministic=True)
            # PPO returns (action, log_prob, value), others return just action
            if isinstance(agent, PPO):
                action = agent_result[0]  # Extract action from tuple
            else:
                action = agent_result
            # Agent returns [phase, action], ensure phase=0 (betting)
            if action[0] != 0:
                action[0] = 0
            bet_size_idx = int(action[1])
            bet_amount = base_bet * bet_multipliers[bet_size_idx]
        else:
            # Determine bet size using strategy
            if bet_strategy == "fixed":
                bet_size_idx = bet_multiplier - 1  # Convert to 0-indexed
                bet_size_idx = max(0, min(bet_size_idx, len(bet_multipliers) - 1))
                bet_amount = base_bet * bet_multipliers[bet_size_idx]
            elif bet_strategy == "random":
                bet_size_idx = np.random.randint(0, len(bet_multipliers))
                bet_amount = base_bet * bet_multipliers[bet_size_idx]
            elif bet_strategy == "proportional":
                # Bet 5% of current bankroll, rounded to nearest bet multiplier
                target_bet = current_bankroll * 0.05
                bet_size_idx = min(
                    len(bet_multipliers) - 1,
                    max(0, int(np.round(target_bet / base_bet)) - 1)
                )
                bet_amount = base_bet * bet_multipliers[bet_size_idx]
            else:
                bet_size_idx = 0
                bet_amount = base_bet * bet_multipliers[0]
            
            # Make sure we don't bet more than we have
            if bet_amount > current_bankroll:
                bet_amount = current_bankroll
                # Find largest bet multiplier we can afford
                for i in range(len(bet_multipliers) - 1, -1, -1):
                    if base_bet * bet_multipliers[i] <= current_bankroll:
                        bet_size_idx = i
                        bet_amount = base_bet * bet_multipliers[i]
                        break
            
            action = np.array([0, bet_size_idx])  # [phase=betting, bet_index]
        
        # Make sure we don't bet more than we have (for agent too)
        if bet_amount > current_bankroll:
            bet_amount = current_bankroll
            # Find largest bet multiplier we can afford
            for i in range(len(bet_multipliers) - 1, -1, -1):
                if base_bet * bet_multipliers[i] <= current_bankroll:
                    bet_size_idx = i
                    bet_amount = base_bet * bet_multipliers[i]
                    action[1] = bet_size_idx
                    break
        
        state, reward, terminated, truncated, info = env.step(action)
        
        # Phase 2: Playing
        done = False
        while not done:
            if use_agent:
                # Use trained agent to make decisions
                agent_result = agent.act(state, deterministic=True)
                # PPO returns (action, log_prob, value), others return just action
                if isinstance(agent, PPO):
                    action = agent_result[0]  # Extract action from tuple
                else:
                    action = agent_result
                # Ensure we're in playing phase
                if action[0] != 1:
                    action[0] = 1
            else:
                # Simple strategy: hit if sum < 17, stand otherwise
                # Check if we can double
                if info.get('can_double', False) and np.random.random() < 0.1:  # 10% chance to double
                    action = np.array([1, 2])  # Double
                else:
                    # Get current player sum from state
                    player_sum = int(state[0])
                    if player_sum < 17:
                        action = np.array([1, 0])  # Hit
                    else:
                        action = np.array([1, 1])  # Stand
            
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        
        # Update bankroll
        current_bankroll += reward
        bankroll_history.append(current_bankroll)
        
        # Track game result
        game_results.append({
            'iteration': iteration + 1,
            'bet_amount': bet_amount,
            'reward': reward,
            'bankroll_after': current_bankroll,
            'won': reward > 0,
        })
        
        # Print progress every 10 iterations
        if (iteration + 1) % 10 == 0:
            print(f"Iteration {iteration + 1}: Bankroll = ${current_bankroll:.2f} "
                  f"(Change: ${current_bankroll - initial_bankroll:+.2f})")
    
    # Final statistics
    final_bankroll = current_bankroll
    total_change = final_bankroll - initial_bankroll
    percent_change = (total_change / initial_bankroll) * 100
    
    wins = sum(1 for r in game_results if r['won'])
    losses = len(game_results) - wins
    
    print("\n" + "=" * 50)
    print("SIMULATION RESULTS")
    print("=" * 50)
    print(f"Initial Bankroll: ${initial_bankroll:.2f}")
    print(f"Final Bankroll: ${final_bankroll:.2f}")
    print(f"Total Change: ${total_change:+.2f} ({percent_change:+.2f}%)")
    print(f"Games Played: {len(game_results)}")
    print(f"Wins: {wins} ({wins/len(game_results)*100:.1f}%)")
    print(f"Losses: {losses} ({losses/len(game_results)*100:.1f}%)")
    print(f"Average Bet: ${np.mean([r['bet_amount'] for r in game_results]):.2f}")
    print(f"Average Reward per Game: ${np.mean([r['reward'] for r in game_results]):.2f}")
    print("=" * 50)
    
    # Plot results
    plt.figure(figsize=(12, 6))
    
    # Plot 1: Bankroll over time
    plt.subplot(1, 2, 1)
    plt.plot(range(len(bankroll_history)), bankroll_history, linewidth=2)
    plt.axhline(y=initial_bankroll, color='r', linestyle='--', label='Initial Bankroll')
    plt.xlabel('Game Number')
    plt.ylabel('Bankroll ($)')
    plt.title('Bankroll Over Time')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 2: Cumulative change
    plt.subplot(1, 2, 2)
    cumulative_change = [b - initial_bankroll for b in bankroll_history]
    plt.plot(range(len(cumulative_change)), cumulative_change, linewidth=2, color='green')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Game Number')
    plt.ylabel('Cumulative Change ($)')
    plt.title('Cumulative Profit/Loss')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_file = 'bankroll_simulation.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {output_file}")
    
    plt.show()
    
    return bankroll_history, game_results


def main():
    parser = argparse.ArgumentParser(description='Simulate bankroll over multiple Blackjack games')
    parser.add_argument('--initial_bankroll', type=float, default=100.0,
                       help='Starting bankroll (default: 100.0)')
    parser.add_argument('--num_iterations', type=int, default=100,
                       help='Number of games to play (default: 100)')
    parser.add_argument('--bet_strategy', type=str, default='fixed',
                       choices=['fixed', 'random', 'proportional'],
                       help='Betting strategy (default: fixed)')
    parser.add_argument('--bet_multiplier', type=int, default=1,
                       choices=[1, 2, 3],
                       help='Bet multiplier for fixed strategy (default: 1)')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to config file')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to trained model file (.pth)')
    parser.add_argument('--algorithm', type=str, default=None,
                       choices=['dqn', 'ddqn', 'dueling_ddqn', 'ppo'],
                       help='Algorithm type (required if --model_path is provided)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.model_path and not args.algorithm:
        parser.error("--algorithm is required when --model_path is provided")
    if args.algorithm and not args.model_path:
        parser.error("--model_path is required when --algorithm is provided")
    
    simulate_bankroll(
        initial_bankroll=args.initial_bankroll,
        num_iterations=args.num_iterations,
        bet_strategy=args.bet_strategy,
        bet_multiplier=args.bet_multiplier,
        config_path=args.config,
        seed=args.seed,
        model_path=args.model_path,
        algorithm=args.algorithm,
    )


if __name__ == '__main__':
    main()

