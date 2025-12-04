"""Training script for multi-objective RL Blackjack"""

import argparse
import yaml
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from pathlib import Path
import json

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.environment import CustomBlackjackEnv
from src.algorithms import DQN, DDQN, DuelingDDQN, PPO
from src.algorithms.ppo import MultiObjectivePPO
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
        # Check if separate critics are enabled
        use_separate_critics = config.get('objectives', {}).get('use_separate_critics', False)
        
        if use_separate_critics:
            objective_names = list(config['objectives']['weights'].keys())
            return MultiObjectivePPO(
                state_dim=state_dim,
                action_dim=action_dim,
                objective_names=objective_names,
                objective_weights=config['objectives']['weights'],
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


def train_episode(env, agent, multi_obj, is_ppo=False, use_separate_critics=False):
    """Train one episode."""
    from src.algorithms.ppo import MultiObjectivePPO
    
    state, info = env.reset()
    episode_reward = 0.0
    episode_return = 0.0
    episode_wins = 0
    episode_games = 0
    done = False
    
    while not done:
        # Get action
        if is_ppo:
            if use_separate_critics:
                action, log_prob, values_dict, combined_value = agent.act(state, deterministic=False)
                # values_dict is already a dict of scalars
                objective_values = values_dict
            else:
                action, log_prob, value = agent.act(state, deterministic=False)
                objective_values = {'combined': value}
                combined_value = value
        else:
            action = agent.act(state, deterministic=False)
            log_prob = 0.0
            value = 0.0
            objective_values = {'combined': value}
            combined_value = value
        
        # Step environment
        next_state, raw_reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Compute multi-objective reward
        game_ended = done
        won = info.get('won', False)
        combined_reward, obj_values = multi_obj.compute_reward(raw_reward, game_ended, won)
        
        # Store transition
        if is_ppo:
            if use_separate_critics:
                # Store separate rewards and values per objective
                agent.store_transition(
                    state, action, obj_values, log_prob, objective_values, done
                )
            else:
                agent.store_transition(state, action, combined_reward, log_prob, combined_value, done)
        else:
            agent.update_replay_buffer(state, action, combined_reward, next_state, done)
        
        # Train step (for off-policy, train every step; for PPO, train at end)
        if not is_ppo:
            agent.train_step()
        
        state = next_state
        episode_reward += combined_reward
        episode_return += raw_reward
        
        if game_ended:
            episode_games += 1
            if won:
                episode_wins += 1
    
    # PPO: train at end of episode
    if is_ppo:
        metrics = agent.train_step()
    else:
        metrics = {}
    
    # Update multi-objective tracker
    multi_obj.update_episode(episode_return, episode_wins, episode_games)
    
    return {
        'episode_reward': episode_reward,
        'episode_return': episode_return,
        'episode_wins': episode_wins,
        'episode_games': episode_games,
        'metrics': metrics,
    }


def evaluate(env, agent, num_episodes: int, multi_obj: MultiObjectiveReward):
    """Evaluate agent."""
    from src.algorithms.ppo import MultiObjectivePPO
    agent.eval()
    is_ppo = isinstance(agent, (PPO, MultiObjectivePPO))
    use_separate_critics = isinstance(agent, MultiObjectivePPO)
    
    episode_rewards = []
    episode_returns = []
    episode_wins = []
    episode_games = []
    
    for _ in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0.0
        episode_return = 0.0
        episode_wins_ep = 0
        episode_games_ep = 0
        done = False
        
        while not done:
            if is_ppo:
                if use_separate_critics:
                    action, _, _, _ = agent.act(state, deterministic=True)
                else:
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
            
            if game_ended:
                episode_games_ep += 1
                if won:
                    episode_wins_ep += 1
        
        episode_rewards.append(episode_reward)
        episode_returns.append(episode_return)
        episode_wins.append(episode_wins_ep)
        episode_games.append(episode_games_ep)
    
    agent.train()
    
    return {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_return': np.mean(episode_returns),
        'std_return': np.std(episode_returns),
        'win_rate': np.sum(episode_wins) / max(np.sum(episode_games), 1),
    }


def save_plots(results_dir: str, training_rewards: list, eval_rewards: list, 
                training_losses: list, multi_obj_metrics: dict, algorithm: str):
    """Save visualization plots."""
    os.makedirs(results_dir, exist_ok=True)
    
    # Training reward vs Episode
    plt.figure(figsize=(10, 6))
    plt.plot(training_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Training Reward')
    plt.title(f'{algorithm.upper()} - Training Reward vs Episode')
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, f'{algorithm}_training_reward.png'))
    plt.close()
    
    # Evaluation reward vs Episode
    if eval_rewards:
        plt.figure(figsize=(10, 6))
        episodes = [i * len(training_rewards) // len(eval_rewards) for i in range(len(eval_rewards))]
        plt.plot(episodes, eval_rewards)
        plt.xlabel('Episode')
        plt.ylabel('Evaluation Reward')
        plt.title(f'{algorithm.upper()} - Evaluation Reward vs Episode')
        plt.grid(True)
        plt.savefig(os.path.join(results_dir, f'{algorithm}_eval_reward.png'))
        plt.close()
    
    # Training loss vs Episode
    if training_losses:
        plt.figure(figsize=(10, 6))
        plt.plot(training_losses)
        plt.xlabel('Episode')
        plt.ylabel('Training Loss')
        plt.title(f'{algorithm.upper()} - Training Loss vs Episode')
        plt.grid(True)
        plt.savefig(os.path.join(results_dir, f'{algorithm}_training_loss.png'))
        plt.close()
    
    # Multi-objective metrics
    if multi_obj_metrics:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        if 'expected_return' in multi_obj_metrics:
            axes[0].plot(multi_obj_metrics['expected_return'])
            axes[0].set_xlabel('Episode')
            axes[0].set_ylabel('Expected Return')
            axes[0].set_title('Expected Return')
            axes[0].grid(True)
        
        if 'sharpe_ratio' in multi_obj_metrics:
            axes[1].plot(multi_obj_metrics['sharpe_ratio'])
            axes[1].set_xlabel('Episode')
            axes[1].set_ylabel('Sharpe Ratio')
            axes[1].set_title('Sharpe Ratio')
            axes[1].grid(True)
        
        if 'win_rate' in multi_obj_metrics:
            axes[2].plot(multi_obj_metrics['win_rate'])
            axes[2].set_xlabel('Episode')
            axes[2].set_ylabel('Win Rate')
            axes[2].set_title('Win Rate')
            axes[2].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'{algorithm}_multi_objective.png'))
        plt.close()


def train_single_seed(algorithm: str, config: dict, device: str, seed: int, base_results_dir: str):
    """Train a single seed and return the results directory path."""
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if hasattr(torch, 'cuda') and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    import random
    random.seed(seed)
    
    # Create results directory with seed in name
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = os.path.join(
        base_results_dir,
        f"{algorithm}_{seed}_{timestamp}"
    )
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Training {algorithm.upper()} with seed {seed}")
    print(f"Results directory: {results_dir}")
    print(f"{'='*60}\n")
    
    # Save config with seed info
    config_copy = config.copy()
    config_copy['training']['seed'] = seed
    with open(os.path.join(results_dir, 'config.json'), 'w') as f:
        json.dump(config_copy, f, indent=2)
    
    # Create environment
    env = CustomBlackjackEnv(
        bet_multipliers=config['environment']['bet_multipliers'],
        base_bet=config['environment']['base_bet'],
        natural=config['environment']['natural'],
        sab=config['environment']['sab'],
    )
    
    # Get state and action dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = 6  # Flattened action space
    
    # Create agent
    agent = create_agent(algorithm, state_dim, action_dim, config, device)
    
    # Create multi-objective reward
    multi_obj = MultiObjectiveReward(
        weights=config['objectives']['weights'],
        rolling_window=config['objectives']['rolling_window'],
        risk_free_rate=config['objectives']['risk_free_rate'],
    )
    
    # Training metrics
    training_rewards = []
    eval_rewards = []
    training_losses = []
    multi_obj_metrics = {
        'expected_return': [],
        'sharpe_ratio': [],
        'win_rate': [],
    }
    
    # Training loop
    num_episodes = config['training']['num_episodes']
    eval_freq = config['training']['eval_freq']
    eval_episodes = config['training']['eval_episodes']
    save_freq = config['training']['save_freq']
    log_freq = config['training']['log_freq']
    
    best_eval_reward = float('-inf')
    is_ppo = isinstance(agent, (PPO, MultiObjectivePPO))
    use_separate_critics = isinstance(agent, MultiObjectivePPO)
    
    if use_separate_critics:
        print(f"Training {algorithm.upper()} with separate critics per objective for {num_episodes} episodes (seed {seed})...")
    else:
        print(f"Training {algorithm.upper()} for {num_episodes} episodes (seed {seed})...")
    
    for episode in range(num_episodes):
        # Train episode
        episode_data = train_episode(env, agent, multi_obj, is_ppo, use_separate_critics)
        
        training_rewards.append(episode_data['episode_reward'])
        if 'loss' in episode_data['metrics']:
            training_losses.append(episode_data['metrics']['loss'])
        
        # Get multi-objective metrics
        metrics = multi_obj.get_metrics()
        multi_obj_metrics['expected_return'].append(metrics['expected_return'])
        multi_obj_metrics['sharpe_ratio'].append(metrics['sharpe_ratio'])
        multi_obj_metrics['win_rate'].append(metrics['win_rate'])
        
        # Evaluate
        if (episode + 1) % eval_freq == 0:
            eval_results = evaluate(env, agent, eval_episodes, multi_obj)
            eval_rewards.append(eval_results['mean_reward'])
            
            print(f"Episode {episode + 1}/{num_episodes} (seed {seed})")
            print(f"  Training Reward: {episode_data['episode_reward']:.2f}")
            print(f"  Eval Reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
            print(f"  Eval Return: {eval_results['mean_return']:.2f} ± {eval_results['std_return']:.2f}")
            print(f"  Win Rate: {eval_results['win_rate']:.2%}")
            
            # Save best model
            if eval_results['mean_reward'] > best_eval_reward:
                best_eval_reward = eval_results['mean_reward']
                agent.save(os.path.join(results_dir, 'best_model.pth'))
                print(f"  Saved best model (eval reward: {best_eval_reward:.2f})")
        
        # Save checkpoint
        if (episode + 1) % save_freq == 0:
            agent.save(os.path.join(results_dir, f'checkpoint_ep{episode+1}.pth'))
        
        # Log
        if (episode + 1) % log_freq == 0:
            print(f"Episode {episode + 1}/{num_episodes} (seed {seed}) - Reward: {episode_data['episode_reward']:.2f}")
    
    # Save final model
    agent.save(os.path.join(results_dir, 'final_model.pth'))
    
    # Save metrics to CSV
    df = pd.DataFrame({
        'episode': range(1, len(training_rewards) + 1),
        'training_reward': training_rewards,
        'training_loss': training_losses + [None] * (len(training_rewards) - len(training_losses)),
        'expected_return': multi_obj_metrics['expected_return'],
        'sharpe_ratio': multi_obj_metrics['sharpe_ratio'],
        'win_rate': multi_obj_metrics['win_rate'],
    })
    df.to_csv(os.path.join(results_dir, 'training_metrics.csv'), index=False)
    
    if eval_rewards:
        eval_df = pd.DataFrame({
            'episode': [(i + 1) * eval_freq for i in range(len(eval_rewards))],
            'eval_reward': eval_rewards,
        })
        eval_df.to_csv(os.path.join(results_dir, 'eval_metrics.csv'), index=False)
    
    # Save plots
    save_plots(results_dir, training_rewards, eval_rewards, training_losses, 
               multi_obj_metrics, algorithm)
    
    # Save summary
    summary = {
        'algorithm': algorithm,
        'seed': seed,
        'total_episodes': num_episodes,
        'final_training_reward': np.mean(training_rewards[-100:]) if training_rewards else 0.0,
        'best_eval_reward': best_eval_reward,
        'final_metrics': multi_obj.get_metrics(),
    }
    
    with open(os.path.join(results_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nTraining complete for seed {seed}! Results saved to {results_dir}")
    print(f"Best evaluation reward: {best_eval_reward:.2f}")
    
    return results_dir, summary


def main():
    parser = argparse.ArgumentParser(description='Train RL agent for Blackjack')
    parser.add_argument('--algorithm', type=str, required=True, 
                       choices=['dqn', 'ddqn', 'dueling_ddqn', 'ppo'],
                       help='Algorithm to train')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to config file')
    parser.add_argument('--results_dir', type=str, default=None,
                       help='Results directory (overrides config)')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Set device
    device = config['training']['device']
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    
    # Get seeds - handle both list and single value
    seeds_config = config['training'].get('seed', 42)
    if isinstance(seeds_config, list):
        seeds = seeds_config
    else:
        seeds = [seeds_config]
    
    # Create base results directory structure: results/{algorithm}/
    if args.results_dir:
        base_results_dir = args.results_dir
    else:
        base_results_dir = os.path.join(
            config['output']['results_dir'],
            args.algorithm
        )
    os.makedirs(base_results_dir, exist_ok=True)
    
    # Train for each seed
    seed_results = []
    for seed in seeds:
        results_dir, summary = train_single_seed(
            args.algorithm, config, device, seed, base_results_dir
        )
        seed_results.append({
            'seed': seed,
            'results_dir': results_dir,
            'summary': summary
        })
    
    # After all seeds are trained, aggregate results
    print(f"\n{'='*60}")
    print(f"All seeds completed! Aggregating results...")
    print(f"{'='*60}\n")
    
    # Import and run aggregation
    try:
        # Import using the same path setup as other modules
        from src.training.aggregate_results import aggregate_seed_results
        aggregate_seed_results(base_results_dir, args.algorithm, seed_results)
    except Exception as e:
        print(f"Warning: Could not aggregate results: {e}")
        print("You can run aggregation separately with:")
        print(f"  python src/training/aggregate_results.py --algorithm {args.algorithm} --results_dir {base_results_dir}")


if __name__ == '__main__':
    main()

