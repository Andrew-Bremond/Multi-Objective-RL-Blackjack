"""Aggregate results across multiple seeds for an algorithm"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict


def load_seed_data(results_dir: str) -> Dict:
    """Load training and evaluation data from a seed's results directory."""
    data = {}
    
    # Load training metrics
    training_csv = os.path.join(results_dir, 'training_metrics.csv')
    if os.path.exists(training_csv):
        data['training'] = pd.read_csv(training_csv)
    
    # Load eval metrics
    eval_csv = os.path.join(results_dir, 'eval_metrics.csv')
    if os.path.exists(eval_csv):
        data['eval'] = pd.read_csv(eval_csv)
    
    # Load summary
    summary_json = os.path.join(results_dir, 'summary.json')
    if os.path.exists(summary_json):
        with open(summary_json, 'r') as f:
            data['summary'] = json.load(f)
    
    return data


def aggregate_seed_results(base_results_dir: str, algorithm: str, seed_results: List[Dict] = None):
    """
    Aggregate results across multiple seeds and create summary plots.
    
    Args:
        base_results_dir: Base directory containing seed subdirectories (e.g., results/dqn/)
        algorithm: Algorithm name
        seed_results: Optional list of seed result dictionaries from training
    """
    # If seed_results not provided, scan directory for seed folders
    if seed_results is None:
        seed_results = []
        for item in os.listdir(base_results_dir):
            item_path = os.path.join(base_results_dir, item)
            if os.path.isdir(item_path) and item.startswith(f"{algorithm}_"):
                # Extract seed from directory name (format: algorithm_seed_timestamp)
                parts = item.split('_')
                if len(parts) >= 2:
                    try:
                        seed = int(parts[1])
                        seed_data = load_seed_data(item_path)
                        if seed_data:
                            seed_results.append({
                                'seed': seed,
                                'results_dir': item_path,
                                'summary': seed_data.get('summary', {})
                            })
                    except ValueError:
                        continue
    
    if not seed_results:
        print(f"No seed results found in {base_results_dir}")
        return
    
    # Sort by seed
    seed_results.sort(key=lambda x: x['seed'])
    
    print(f"Aggregating results for {algorithm.upper()} across {len(seed_results)} seeds...")
    
    # Load all seed data
    all_training_data = []
    all_eval_data = []
    all_summaries = []
    
    for seed_result in seed_results:
        seed_data = load_seed_data(seed_result['results_dir'])
        if 'training' in seed_data:
            df = seed_data['training'].copy()
            df['seed'] = seed_result['seed']
            all_training_data.append(df)
        if 'eval' in seed_data:
            df = seed_data['eval'].copy()
            df['seed'] = seed_result['seed']
            all_eval_data.append(df)
        if 'summary' in seed_data:
            all_summaries.append(seed_data['summary'])
    
    if not all_training_data:
        print("No training data found to aggregate")
        return
    
    # Combine training data
    training_df = pd.concat(all_training_data, ignore_index=True)
    
    # Combine eval data
    eval_df = pd.concat(all_eval_data, ignore_index=True) if all_eval_data else None
    
    # Create aggregated plots
    create_aggregated_plots(
        base_results_dir, algorithm, training_df, eval_df, seed_results
    )
    
    # Find and save best model info
    find_best_model(base_results_dir, algorithm, seed_results, all_summaries)


def create_aggregated_plots(
    base_results_dir: str,
    algorithm: str,
    training_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    seed_results: List[Dict]
):
    """Create averaged plots across seeds."""
    
    # 1. Training Reward (averaged across seeds)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Training Reward
    ax = axes[0, 0]
    seeds = training_df['seed'].unique()
    for seed in seeds:
        seed_data = training_df[training_df['seed'] == seed]
        ax.plot(seed_data['episode'], seed_data['training_reward'], 
                alpha=0.3, label=f'Seed {seed}')
    
    # Average across seeds
    avg_training = training_df.groupby('episode')['training_reward'].mean()
    std_training = training_df.groupby('episode')['training_reward'].std()
    ax.plot(avg_training.index, avg_training.values, 'b-', linewidth=2, 
            label='Average', zorder=10)
    ax.fill_between(avg_training.index, 
                    avg_training.values - std_training.values,
                    avg_training.values + std_training.values,
                    alpha=0.2, color='blue')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Training Reward')
    ax.set_title(f'{algorithm.upper()} - Training Reward (Averaged Across Seeds)')
    ax.legend()
    ax.grid(True)
    
    # Eval Reward
    ax = axes[0, 1]
    if eval_df is not None and len(eval_df) > 0:
        seeds = eval_df['seed'].unique()
        for seed in seeds:
            seed_data = eval_df[eval_df['seed'] == seed]
            ax.plot(seed_data['episode'], seed_data['eval_reward'], 
                    alpha=0.3, label=f'Seed {seed}')
        
        # Average across seeds
        avg_eval = eval_df.groupby('episode')['eval_reward'].mean()
        std_eval = eval_df.groupby('episode')['eval_reward'].std()
        ax.plot(avg_eval.index, avg_eval.values, 'r-', linewidth=2, 
                label='Average', zorder=10)
        ax.fill_between(avg_eval.index,
                        avg_eval.values - std_eval.values,
                        avg_eval.values + std_eval.values,
                        alpha=0.2, color='red')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Evaluation Reward')
        ax.set_title(f'{algorithm.upper()} - Evaluation Reward (Averaged Across Seeds)')
        ax.legend()
        ax.grid(True)
    else:
        ax.text(0.5, 0.5, 'No evaluation data available', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'{algorithm.upper()} - Evaluation Reward')
    
    # Expected Return
    ax = axes[1, 0]
    seeds = training_df['seed'].unique()
    for seed in seeds:
        seed_data = training_df[training_df['seed'] == seed]
        ax.plot(seed_data['episode'], seed_data['expected_return'], 
                alpha=0.3, label=f'Seed {seed}')
    
    avg_er = training_df.groupby('episode')['expected_return'].mean()
    std_er = training_df.groupby('episode')['expected_return'].std()
    ax.plot(avg_er.index, avg_er.values, 'g-', linewidth=2, 
            label='Average', zorder=10)
    ax.fill_between(avg_er.index,
                    avg_er.values - std_er.values,
                    avg_er.values + std_er.values,
                    alpha=0.2, color='green')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Expected Return')
    ax.set_title(f'{algorithm.upper()} - Expected Return (Averaged Across Seeds)')
    ax.legend()
    ax.grid(True)
    
    # Sharpe Ratio and Win Rate
    ax = axes[1, 1]
    # Sharpe Ratio
    seeds = training_df['seed'].unique()
    for seed in seeds:
        seed_data = training_df[training_df['seed'] == seed]
        ax.plot(seed_data['episode'], seed_data['sharpe_ratio'], 
                alpha=0.3, linestyle='--', label=f'Sharpe (Seed {seed})')
    
    avg_sr = training_df.groupby('episode')['sharpe_ratio'].mean()
    std_sr = training_df.groupby('episode')['sharpe_ratio'].std()
    ax.plot(avg_sr.index, avg_sr.values, 'orange', linewidth=2, 
            linestyle='--', label='Sharpe (Avg)', zorder=10)
    
    # Win Rate
    for seed in seeds:
        seed_data = training_df[training_df['seed'] == seed]
        ax.plot(seed_data['episode'], seed_data['win_rate'], 
                alpha=0.3, linestyle=':', label=f'Win Rate (Seed {seed})')
    
    avg_wr = training_df.groupby('episode')['win_rate'].mean()
    std_wr = training_df.groupby('episode')['win_rate'].std()
    ax.plot(avg_wr.index, avg_wr.values, 'purple', linewidth=2, 
            linestyle=':', label='Win Rate (Avg)', zorder=10)
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Value')
    ax.set_title(f'{algorithm.upper()} - Sharpe Ratio & Win Rate (Averaged)')
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(base_results_dir, f'{algorithm}_aggregated_results.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved aggregated plot to {plot_path}")
    
    # Create separate multi-objective plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    metrics = ['expected_return', 'sharpe_ratio', 'win_rate']
    titles = ['Expected Return', 'Sharpe Ratio', 'Win Rate']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx]
        seeds = training_df['seed'].unique()
        for seed in seeds:
            seed_data = training_df[training_df['seed'] == seed]
            ax.plot(seed_data['episode'], seed_data[metric], 
                    alpha=0.3, label=f'Seed {seed}')
        
        avg = training_df.groupby('episode')[metric].mean()
        std = training_df.groupby('episode')[metric].std()
        ax.plot(avg.index, avg.values, linewidth=2, label='Average', zorder=10)
        ax.fill_between(avg.index,
                        avg.values - std.values,
                        avg.values + std.values,
                        alpha=0.2)
        ax.set_xlabel('Episode')
        ax.set_ylabel(title)
        ax.set_title(f'{algorithm.upper()} - {title} (Averaged Across Seeds)')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(base_results_dir, f'{algorithm}_multi_objective_aggregated.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved multi-objective aggregated plot to {plot_path}")


def find_best_model(
    base_results_dir: str,
    algorithm: str,
    seed_results: List[Dict],
    all_summaries: List[Dict]
):
    """Find the best model across all seeds and save summary."""
    
    if not all_summaries:
        print("No summaries found to determine best model")
        return
    
    # Find best by eval reward
    best_eval_reward = float('-inf')
    best_seed = None
    best_summary = None
    best_dir = None
    
    for i, summary in enumerate(all_summaries):
        eval_reward = summary.get('best_eval_reward', float('-inf'))
        if eval_reward > best_eval_reward:
            best_eval_reward = eval_reward
            best_seed = seed_results[i]['seed']
            best_summary = summary
            best_dir = seed_results[i]['results_dir']
    
    # Create summary text
    summary_text = f"""
{'='*60}
Best Model Summary for {algorithm.upper()}
{'='*60}

Best Model (by Evaluation Reward):
  Seed: {best_seed}
  Best Eval Reward: {best_eval_reward:.4f}
  Results Directory: {best_dir}
  
  Final Metrics:
    Expected Return: {best_summary.get('final_metrics', {}).get('expected_return', 'N/A'):.4f}
    Sharpe Ratio: {best_summary.get('final_metrics', {}).get('sharpe_ratio', 'N/A'):.4f}
    Win Rate: {best_summary.get('final_metrics', {}).get('win_rate', 'N/A'):.4f}
  
  Final Training Reward (last 100 episodes): {best_summary.get('final_training_reward', 'N/A'):.4f}
  Total Episodes: {best_summary.get('total_episodes', 'N/A')}

{'='*60}
All Seeds Summary:
{'='*60}
"""
    
    for i, (seed_result, summary) in enumerate(zip(seed_results, all_summaries)):
        summary_text += f"""
Seed {seed_result['seed']}:
  Best Eval Reward: {summary.get('best_eval_reward', 'N/A'):.4f}
  Final Training Reward: {summary.get('final_training_reward', 'N/A'):.4f}
  Results Directory: {seed_result['results_dir']}
"""
    
    summary_text += f"\n{'='*60}\n"
    
    # Save to file
    summary_path = os.path.join(base_results_dir, f'{algorithm}_best_model_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(summary_text)
    
    print(f"\nBest model summary saved to {summary_path}")
    print(summary_text)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Aggregate results across seeds')
    parser.add_argument('--algorithm', type=str, required=True,
                       choices=['dqn', 'ddqn', 'dueling_ddqn', 'ppo'],
                       help='Algorithm name')
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Base results directory (e.g., results/dqn/)')
    
    args = parser.parse_args()
    
    aggregate_seed_results(args.results_dir, args.algorithm)


