"""Evaluation script for trained RL agents"""

import argparse
import yaml
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.environment import CustomBlackjackEnv
from src.algorithms import DQN, DDQN, DuelingDDQN, PPO
from src.objectives import MultiObjectiveReward


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


def evaluate_agent(env, agent, num_episodes: int, multi_obj: MultiObjectiveReward):
    """Evaluate agent and return detailed metrics."""
    agent.eval()
    is_ppo = isinstance(agent, PPO)
    
    episode_rewards = []
    episode_returns = []
    episode_wins = []
    episode_games = []
    episode_lengths = []
    
    # Per-game metrics
    game_returns = []
    game_wins = []
    game_bets = []
    
    for _ in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0.0
        episode_return = 0.0
        episode_wins_ep = 0
        episode_games_ep = 0
        episode_length = 0
        done = False
        
        while not done:
            if is_ppo:
                action, _, _ = agent.act(state, deterministic=True)
            else:
                action = agent.act(state, deterministic=True)
            
            next_state, raw_reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            game_ended = done
            won = info.get('won', False)
            combined_reward, _ = multi_obj.compute_reward(raw_reward, game_ended, won)
            
            state = next_state
            episode_reward += combined_reward
            episode_return += raw_reward
            episode_length += 1
            
            if game_ended:
                episode_games_ep += 1
                game_returns.append(raw_reward)
                game_wins.append(1 if won else 0)
                game_bets.append(info.get('bet_amount', 0))
                if won:
                    episode_wins_ep += 1
        
        episode_rewards.append(episode_reward)
        episode_returns.append(episode_return)
        episode_wins.append(episode_wins_ep)
        episode_games.append(episode_games_ep)
        episode_lengths.append(episode_length)
    
    agent.train()
    
    # Compute statistics
    metrics = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_return': np.mean(episode_returns),
        'std_return': np.std(episode_returns),
        'mean_episode_length': np.mean(episode_lengths),
        'win_rate': np.sum(episode_wins) / max(np.sum(episode_games), 1),
        'total_games': np.sum(episode_games),
        'total_wins': np.sum(episode_wins),
        'mean_game_return': np.mean(game_returns) if game_returns else 0.0,
        'std_game_return': np.std(game_returns) if game_returns else 0.0,
        'mean_bet': np.mean(game_bets) if game_bets else 0.0,
    }
    
    # Compute Sharpe ratio
    if len(game_returns) > 1:
        mean_return = np.mean(game_returns)
        std_return = np.std(game_returns)
        if std_return > 0:
            metrics['sharpe_ratio'] = mean_return / std_return
        else:
            metrics['sharpe_ratio'] = 0.0
    else:
        metrics['sharpe_ratio'] = 0.0
    
    return metrics


def compare_algorithms(model_paths: dict, config_path: str, num_episodes: int = 1000):
    """Compare multiple algorithms."""
    config = load_config(config_path)
    
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
    
    state_dim = env.observation_space.shape[0]
    action_dim = 6
    
    # Create multi-objective reward
    multi_obj = MultiObjectiveReward(
        weights=config['objectives']['weights'],
        rolling_window=config['objectives']['rolling_window'],
        risk_free_rate=config['objectives']['risk_free_rate'],
    )
    
    results = {}
    
    print("Evaluating algorithms...")
    for algorithm, model_path in model_paths.items():
        print(f"\n{algorithm.upper()}:")
        
        # Create agent
        agent = create_agent(algorithm, state_dim, action_dim, config, device)
        
        # Load model
        agent.load(model_path)
        
        # Evaluate
        metrics = evaluate_agent(env, agent, num_episodes, multi_obj)
        results[algorithm] = metrics
        
        # Print results
        print(f"  Mean Reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
        print(f"  Mean Return: {metrics['mean_return']:.2f} ± {metrics['std_return']:.2f}")
        print(f"  Win Rate: {metrics['win_rate']:.2%}")
        print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"  Mean Game Return: {metrics['mean_game_return']:.2f} ± {metrics['std_game_return']:.2f}")
    
    # Create comparison plot
    algorithms = list(results.keys())
    metrics_to_plot = ['mean_reward', 'win_rate', 'sharpe_ratio']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, metric in enumerate(metrics_to_plot):
        values = [results[alg][metric] for alg in algorithms]
        axes[i].bar(algorithms, values)
        axes[i].set_title(metric.replace('_', ' ').title())
        axes[i].set_ylabel('Value')
        axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('algorithm_comparison.png', dpi=300, bbox_inches='tight')
    print("\nComparison plot saved to algorithm_comparison.png")
    
    # Save results to CSV
    df = pd.DataFrame(results).T
    df.to_csv('algorithm_comparison.csv')
    print("Comparison results saved to algorithm_comparison.csv")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained RL agent')
    parser.add_argument('--model_path', type=str, required=False,
                       help='Path to model file')
    parser.add_argument('--algorithm', type=str, required=False,
                       choices=['dqn', 'ddqn', 'dueling_ddqn', 'ppo'],
                       help='Algorithm type')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to config file')
    parser.add_argument('--num_episodes', type=int, default=1000,
                       help='Number of episodes for evaluation')
    parser.add_argument('--compare', action='store_true',
                       help='Compare multiple algorithms (requires --model_paths)')
    parser.add_argument('--model_paths', type=str, nargs='+',
                       help='Paths to models for comparison (format: algorithm:path)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.compare:
        if not args.model_paths:
            parser.error("--compare requires --model_paths")
    else:
        if not args.model_path:
            parser.error("--model_path is required when not using --compare")
        if not args.algorithm:
            parser.error("--algorithm is required when not using --compare")
    
    if args.compare and args.model_paths:
        # Parse model paths
        model_paths = {}
        for path_str in args.model_paths:
            parts = path_str.split(':')
            if len(parts) == 2:
                algorithm, path = parts
                model_paths[algorithm] = path
            else:
                print(f"Invalid format: {path_str}. Expected 'algorithm:path'")
                return
        
        compare_algorithms(model_paths, args.config, args.num_episodes)
    else:
        # Single agent evaluation
        config = load_config(args.config)
        
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
        
        state_dim = env.observation_space.shape[0]
        action_dim = 6
        
        # Create agent
        agent = create_agent(args.algorithm, state_dim, action_dim, config, device)
        
        # Load model
        agent.load(args.model_path)
        
        # Create multi-objective reward
        multi_obj = MultiObjectiveReward(
            weights=config['objectives']['weights'],
            rolling_window=config['objectives']['rolling_window'],
            risk_free_rate=config['objectives']['risk_free_rate'],
        )
        
        # Evaluate
        print(f"Evaluating {args.algorithm.upper()} for {args.num_episodes} episodes...")
        metrics = evaluate_agent(env, agent, args.num_episodes, multi_obj)
        
        # Print results
        print("\nEvaluation Results:")
        print(f"  Mean Reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
        print(f"  Mean Return: {metrics['mean_return']:.2f} ± {metrics['std_return']:.2f}")
        print(f"  Win Rate: {metrics['win_rate']:.2%}")
        print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"  Mean Game Return: {metrics['mean_game_return']:.2f} ± {metrics['std_game_return']:.2f}")
        print(f"  Total Games: {metrics['total_games']}")
        print(f"  Total Wins: {metrics['total_wins']}")
        print(f"  Mean Episode Length: {metrics['mean_episode_length']:.2f}")


if __name__ == '__main__':
    main()

