#!/usr/bin/env python3
"""
Interactive GUI for watching a trained RL agent play Blackjack.
Shows real-time game state, bankroll, and actions.
"""

import argparse
import yaml
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from typing import Optional

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.environment import CustomBlackjackEnv
from src.algorithms import DQN, DDQN, DuelingDDQN, PPO
from src.algorithms.ppo import MultiObjectivePPO
import torch


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def detect_model_type(model_path: str) -> tuple:
    """
    Detect if model was saved with separate critics by checking checkpoint structure.
    
    Returns:
        (use_separate_critics, objective_names, objective_weights)
    """
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        state_dict = checkpoint.get('actor_critic', {})
        
        # Check if state_dict has separate critics structure
        has_separate_critics = any('critics.' in key for key in state_dict.keys())
        
        if has_separate_critics:
            # Extract objective names from state dict keys
            objective_names = set()
            for key in state_dict.keys():
                if 'critics.' in key:
                    # Extract objective name from key like "critics.expected_return.0.weight"
                    parts = key.split('.')
                    if len(parts) >= 2:
                        objective_names.add(parts[1])
            objective_names = sorted(list(objective_names))
            
            # Get objective weights if available
            objective_weights = checkpoint.get('objective_weights', None)
            
            return True, objective_names, objective_weights
        else:
            return False, None, None
    except Exception as e:
        # If we can't detect, assume standard PPO
        print(f"Warning: Could not detect model type, assuming standard PPO: {e}")
        return False, None, None


def create_agent(algorithm: str, state_dim: int, action_dim: int, config: dict, device: str, 
                 use_separate_critics: bool = False, objective_names: list = None, 
                 objective_weights: dict = None):
    """Create agent based on algorithm name."""
    algo_config = config['algorithms'][algorithm]
    
    if algorithm == 'dqn':
        return DQN(
            state_dim=state_dim, action_dim=action_dim,
            lr=algo_config['lr'], gamma=algo_config['gamma'],
            epsilon_start=algo_config['epsilon_start'], epsilon_end=algo_config['epsilon_end'],
            epsilon_decay=algo_config['epsilon_decay'], batch_size=algo_config['batch_size'],
            buffer_size=algo_config['buffer_size'], target_update_freq=algo_config['target_update_freq'],
            device=device, hidden_dims=algo_config['hidden_dims'],
        )
    elif algorithm == 'ddqn':
        return DDQN(
            state_dim=state_dim, action_dim=action_dim,
            lr=algo_config['lr'], gamma=algo_config['gamma'],
            epsilon_start=algo_config['epsilon_start'], epsilon_end=algo_config['epsilon_end'],
            epsilon_decay=algo_config['epsilon_decay'], batch_size=algo_config['batch_size'],
            buffer_size=algo_config['buffer_size'], target_update_freq=algo_config['target_update_freq'],
            device=device, hidden_dims=algo_config['hidden_dims'],
        )
    elif algorithm == 'dueling_ddqn':
        return DuelingDDQN(
            state_dim=state_dim, action_dim=action_dim,
            lr=algo_config['lr'], gamma=algo_config['gamma'],
            epsilon_start=algo_config['epsilon_start'], epsilon_end=algo_config['epsilon_end'],
            epsilon_decay=algo_config['epsilon_decay'], batch_size=algo_config['batch_size'],
            buffer_size=algo_config['buffer_size'], target_update_freq=algo_config['target_update_freq'],
            device=device, hidden_dims=algo_config['hidden_dims'],
        )
    elif algorithm == 'ppo':
        if use_separate_critics:
            if objective_names is None:
                objective_names = list(config['objectives']['weights'].keys())
            if objective_weights is None:
                objective_weights = config['objectives']['weights']
            return MultiObjectivePPO(
                state_dim=state_dim, action_dim=action_dim,
                objective_names=objective_names,
                objective_weights=objective_weights,
                lr=algo_config['lr'], gamma=algo_config['gamma'],
                gae_lambda=algo_config['gae_lambda'], clip_epsilon=algo_config['clip_epsilon'],
                value_coef=algo_config['value_coef'], entropy_coef=algo_config['entropy_coef'],
                max_grad_norm=algo_config['max_grad_norm'], ppo_epochs=algo_config['ppo_epochs'],
                batch_size=algo_config['batch_size'], device=device, hidden_dims=algo_config['hidden_dims'],
            )
        else:
            return PPO(
                state_dim=state_dim, action_dim=action_dim,
                lr=algo_config['lr'], gamma=algo_config['gamma'],
                gae_lambda=algo_config['gae_lambda'], clip_epsilon=algo_config['clip_epsilon'],
                value_coef=algo_config['value_coef'], entropy_coef=algo_config['entropy_coef'],
                max_grad_norm=algo_config['max_grad_norm'], ppo_epochs=algo_config['ppo_epochs'],
                batch_size=algo_config['batch_size'], device=device, hidden_dims=algo_config['hidden_dims'],
            )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def get_card_value(card_num):
    """Convert card number to display value."""
    if card_num == 1:
        return "A"
    elif card_num == 11:
        return "J"
    elif card_num == 12:
        return "Q"
    elif card_num == 13:
        return "K"
    else:
        return str(card_num)


def update_gui(fig, ax, bankroll, game_num, player_sum, dealer_card, usable_ace, 
               bet_multiplier, phase, action, reward, game_history, dealer_final_sum=None):
    """Update the GUI display."""
    ax.clear()
    
    # Set up the figure
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 9.5, f'Blackjack - Game #{game_num}', ha='center', va='top', 
            fontsize=16, fontweight='bold')
    
    # Bankroll display (large and prominent)
    bankroll_color = 'green' if bankroll >= 100 else 'red'
    ax.text(5, 8.5, f'Bankroll: ${bankroll:.2f}', ha='center', va='top',
            fontsize=20, fontweight='bold', color=bankroll_color,
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    
    # Game state
    y_pos = 7
    ax.text(1, y_pos, 'Player:', fontsize=12, fontweight='bold')
    # Check if cards haven't been dealt yet (betting phase)
    if phase == 0 and player_sum == 0 and dealer_card == 0:
        player_text = 'Cards not dealt yet'
        ax.text(3, y_pos, player_text, fontsize=12, style='italic', color='gray')
    else:
        player_text = f'Sum: {player_sum}'
        if usable_ace:
            player_text += ' (Ace)'
        ax.text(3, y_pos, player_text, fontsize=12)
    
    ax.text(1, y_pos - 0.8, 'Dealer:', fontsize=12, fontweight='bold')
    # Check if cards haven't been dealt yet
    if phase == 0 and player_sum == 0 and dealer_card == 0:
        ax.text(3, y_pos - 0.8, 'Cards not dealt yet', fontsize=12, style='italic', color='gray')
    else:
        # Show dealer's visible card during play, or final sum if game ended
        if dealer_final_sum is not None:
            ax.text(3, y_pos - 0.8, f'Final Sum: {dealer_final_sum}', fontsize=12, fontweight='bold', color='blue')
        else:
            ax.text(3, y_pos - 0.8, f'Showing: {dealer_card}', fontsize=12)
    
    # Betting info
    if bet_multiplier:
        ax.text(1, y_pos - 1.6, 'Bet:', fontsize=12, fontweight='bold')
        ax.text(3, y_pos - 1.6, f'{bet_multiplier}x (${bet_multiplier:.2f})', fontsize=12)
    
    # Phase and action
    phase_text = 'BETTING' if phase == 0 else 'PLAYING'
    ax.text(1, y_pos - 2.4, 'Phase:', fontsize=12, fontweight='bold')
    ax.text(3, y_pos - 2.4, phase_text, fontsize=12, 
            color='blue' if phase == 0 else 'green')
    
    if action is not None:
        action_names = {
            (0, 0): 'Bet 1x', (0, 1): 'Bet 2x', (0, 2): 'Bet 3x',
            (1, 0): 'HIT', (1, 1): 'STAND', (1, 2): 'DOUBLE'
        }
        action_text = action_names.get(tuple(action), f'Action: {action}')
        ax.text(1, y_pos - 3.2, 'Action:', fontsize=12, fontweight='bold')
        ax.text(3, y_pos - 3.2, action_text, fontsize=12, color='purple')
    
    # Last game result
    if reward is not None and reward != 0:
        result_text = f'Won ${reward:.2f}!' if reward > 0 else f'Lost ${abs(reward):.2f}'
        result_color = 'green' if reward > 0 else 'red'
        ax.text(5, y_pos - 4, result_text, ha='center', fontsize=14, 
                fontweight='bold', color=result_color)
    
    # Game history (last 5 games)
    ax.text(7, y_pos, 'Recent Games:', fontsize=10, fontweight='bold')
    for i, (game, result) in enumerate(game_history[-5:]):
        result_symbol = '✓' if result > 0 else '✗' if result < 0 else '='
        result_color = 'green' if result > 0 else 'red' if result < 0 else 'gray'
        ax.text(7, y_pos - 0.6 - i * 0.4, f'Game {game}: {result_symbol} ${result:+.2f}',
                fontsize=9, color=result_color)
    
    # Instructions
    ax.text(5, 0.5, 'Press Enter to continue, Q to quit', ha='center', 
            fontsize=10, style='italic', color='gray')
    
    plt.tight_layout()
    plt.draw()
    plt.pause(0.1)


def play_with_gui(model_path: str, algorithm: Optional[str] = None, initial_bankroll: float = 100.0,
                  config_path: str = "configs/config.yaml", auto_play: bool = False,
                  delay: float = 1.0):
    """Play Blackjack with GUI visualization."""
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
        render_mode="human"
    )
    
    state_dim = env.observation_space.shape[0]
    action_dim = 6
    
    # Auto-detect if directory is passed instead of file
    if os.path.isdir(model_path):
        model_path = os.path.join(model_path, 'best_model.pth')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"best_model.pth not found in {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Auto-detect algorithm from path if not provided
    if algorithm is None or algorithm == 'auto':
        path_lower = model_path.lower()
        if 'dueling_ddqn' in path_lower or 'dueling-ddqn' in path_lower:
            algorithm = 'dueling_ddqn'
        elif 'ddqn' in path_lower and 'dueling' not in path_lower:
            algorithm = 'ddqn'
        elif 'dqn' in path_lower and 'ddqn' not in path_lower and 'dueling' not in path_lower:
            algorithm = 'dqn'
        elif 'ppo' in path_lower:
            algorithm = 'ppo'
        else:
            raise ValueError(f"Could not auto-detect algorithm from path: {model_path}. Please specify --algorithm")
    
    # Detect if model uses separate critics (for PPO)
    use_separate_critics = False
    objective_names = None
    objective_weights = None
    if algorithm == 'ppo':
        use_separate_critics, objective_names, objective_weights = detect_model_type(model_path)
        if use_separate_critics:
            print(f"Detected MultiObjectivePPO model with objectives: {objective_names}")
    
    # Load agent
    agent = create_agent(algorithm, state_dim, action_dim, config, device, 
                        use_separate_critics, objective_names, objective_weights)
    agent.load(model_path)
    agent.eval()
    is_ppo = isinstance(agent, (PPO, MultiObjectivePPO))
    is_multi_obj_ppo = isinstance(agent, MultiObjectivePPO)
    
    print(f"Loaded {algorithm.upper()} model from {model_path}")
    print("Starting GUI...")
    
    # Set up GUI
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.suptitle('RL Blackjack Agent - Interactive Display', fontsize=14, fontweight='bold')
    
    bankroll = initial_bankroll
    game_num = 0
    game_history = []
    
    try:
        while True:
            game_num += 1
            state, info = env.reset()
            player_sum = int(state[0])
            dealer_card = int(state[1])
            usable_ace = bool(state[2])
            bet_multiplier = info.get('bet_multiplier')
            phase = int(state[4])  # Read phase from state
            action = None
            game_reward = 0.0
            done = False
            
            # Game loop
            while not done:
                # Update GUI
                update_gui(fig, ax, bankroll, game_num, player_sum, dealer_card, usable_ace,
                          bet_multiplier, phase, action, None, game_history, None)
                
                if not auto_play:
                    user_input = input("\nPress Enter to continue (Q to quit): ").strip().lower()
                    if user_input == 'q':
                        raise KeyboardInterrupt
                else:
                    time.sleep(delay)
                
                # Get action from agent
                agent_result = agent.act(state, deterministic=True)
                if is_ppo:
                    if is_multi_obj_ppo:
                        # MultiObjectivePPO returns (action, log_prob, values_dict, combined_value)
                        action = agent_result[0]
                    else:
                        # Standard PPO returns (action, log_prob, value)
                        action = agent_result[0]
                else:
                    action = agent_result
                
                # Step environment
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # Update state
                dealer_final_sum = None
                if not done:
                    player_sum = int(next_state[0])
                    dealer_card = int(next_state[1])
                    usable_ace = bool(next_state[2])
                    phase = int(next_state[4])
                    bet_multiplier = info.get('bet_multiplier')
                else:
                    # Game ended
                    game_reward = reward
                    bankroll += reward
                    game_history.append((game_num, reward))
                    
                    # Get final state
                    player_sum = int(next_state[0])
                    dealer_card = int(next_state[1])
                    usable_ace = bool(next_state[2])
                    phase = int(next_state[4])
                    bet_multiplier = info.get('bet_multiplier')
                    
                    # Get dealer's final sum from info if available
                    dealer_final_sum = info.get('dealer_final_sum')
                
                state = next_state
                
                # Final update after game ends
                if done:
                    update_gui(fig, ax, bankroll, game_num, player_sum, dealer_card, usable_ace,
                              bet_multiplier, phase, action, game_reward, game_history, dealer_final_sum)
                    
                    if not auto_play:
                        user_input = input(f"\nGame ended. Reward: ${game_reward:.2f}. Press Enter to continue (Q to quit): ").strip().lower()
                        if user_input == 'q':
                            raise KeyboardInterrupt
                    else:
                        time.sleep(delay * 2)  # Longer pause after game ends
    
    except KeyboardInterrupt:
        print(f"\n\nFinal Bankroll: ${bankroll:.2f}")
        print(f"Games Played: {game_num}")
        print(f"Total Change: ${bankroll - initial_bankroll:+.2f}")
    
    finally:
        env.close()
        plt.ioff()
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Interactive GUI for RL Blackjack agent')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model file (.pth)')
    parser.add_argument('--algorithm', type=str, required=False, default='auto',
                       choices=['dqn', 'ddqn', 'dueling_ddqn', 'ppo', 'auto'],
                       help='Algorithm type (default: auto-detect from path)')
    parser.add_argument('--initial_bankroll', type=float, default=100.0,
                       help='Starting bankroll (default: 100.0)')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to config file')
    parser.add_argument('--auto', action='store_true',
                       help='Auto-play mode (no user input required)')
    parser.add_argument('--delay', type=float, default=1.0,
                       help='Delay between actions in auto mode (seconds)')
    
    args = parser.parse_args()
    
    # Convert 'auto' to None for auto-detection
    algorithm = None if args.algorithm == 'auto' else args.algorithm
    
    play_with_gui(
        model_path=args.model_path,
        algorithm=algorithm,
        initial_bankroll=args.initial_bankroll,
        config_path=args.config,
        auto_play=args.auto,
        delay=args.delay
    )


if __name__ == '__main__':
    main()