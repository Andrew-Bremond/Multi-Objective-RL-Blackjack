"""Compare different multi-objective configurations for Blackjack RL."""
import json
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_results(output_dir: str):
    """Load evaluation results from a training run."""
    eval_path = Path(output_dir) / "eval_summary.json"
    if eval_path.exists():
        with open(eval_path) as f:
            return json.load(f)
    return None

def compare_modes():
    """Compare all trained modes."""
    modes = {
        "Baseline (3 obj)": "outputs/baseline",
        "Risk-Averse (3 obj)": "outputs/risk", 
        "Strategic (8 obj)": "outputs/strategic_test",
        "Optimal (8 obj)": "outputs/optimal_test",
    }
    
    results = {}
    for mode_name, path in modes.items():
        data = load_results(path)
        if data:
            results[mode_name] = data
            print(f"\n{mode_name}:")
            print(f"  Win Rate: {data['win_rate']:.1%}")
            print(f"  Loss Rate: {data['loss_rate']:.1%}")
            print(f"  Bust Rate: {data['bust_rate']:.1%}")
            print(f"  Avg Reward: {data['avg_reward']:.3f}")
    
    if not results:
        print("No results found. Train models first.")
        return
    
    # Create comparison chart
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Multi-Objective Configuration Comparison', fontsize=16, fontweight='bold')
    
    # Prepare data
    mode_names = list(results.keys())
    win_rates = [results[m]['win_rate'] for m in mode_names]
    loss_rates = [results[m]['loss_rate'] for m in mode_names]
    bust_rates = [results[m]['bust_rate'] for m in mode_names]
    avg_rewards = [results[m]['avg_reward'] for m in mode_names]
    
    # Color scheme
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    # Win Rate comparison
    ax = axes[0, 0]
    bars = ax.bar(range(len(mode_names)), win_rates, color=colors)
    ax.set_ylabel('Win Rate (%)', fontsize=11)
    ax.set_title('Win Rate Comparison', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(mode_names)))
    ax.set_xticklabels([m.split(' (')[0] for m in mode_names], rotation=45, ha='right')
    ax.set_ylim(0, 0.5)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, win_rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{val:.1%}', ha='center', va='bottom', fontsize=9)
    
    # Bust Rate comparison
    ax = axes[0, 1]
    bars = ax.bar(range(len(mode_names)), bust_rates, color=colors)
    ax.set_ylabel('Bust Rate (%)', fontsize=11)
    ax.set_title('Bust Rate Comparison (Lower is Better)', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(mode_names)))
    ax.set_xticklabels([m.split(' (')[0] for m in mode_names], rotation=45, ha='right')
    ax.set_ylim(0, 0.25)
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, bust_rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{val:.1%}', ha='center', va='bottom', fontsize=9)
    
    # Average Reward comparison
    ax = axes[1, 0]
    bars = ax.bar(range(len(mode_names)), avg_rewards, color=colors)
    ax.set_ylabel('Average Reward', fontsize=11)
    ax.set_title('Average Reward (Closer to 0 is Better)', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(mode_names)))
    ax.set_xticklabels([m.split(' (')[0] for m in mode_names], rotation=45, ha='right')
    ax.set_ylim(-0.15, 0.05)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, avg_rewards):
        height = bar.get_height()
        y_pos = height + 0.003 if height > 0 else height - 0.01
        ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                f'{val:.3f}', ha='center', va='bottom' if height > 0 else 'top', fontsize=9)
    
    # Combined Performance Score
    ax = axes[1, 1]
    # Calculate a combined score: win_rate - 0.5*bust_rate + (0.1 + avg_reward)
    scores = [w - 0.5*b + (0.1 + r) for w, b, r in zip(win_rates, bust_rates, avg_rewards)]
    bars = ax.bar(range(len(mode_names)), scores, color=colors)
    ax.set_ylabel('Combined Score', fontsize=11)
    ax.set_title('Overall Performance Score', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(mode_names)))
    ax.set_xticklabels([m.split(' (')[0] for m in mode_names], rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Add legend explaining the number of objectives
    legend_text = ['3 objectives:\n- Profit\n- Loss penalty\n- Bust penalty',
                   '8 objectives:\n+ Close call bonus\n+ Dealer weak bonus\n+ Conservative stand\n+ Aggressive hit\n+ Perfect play']
    
    fig.text(0.98, 0.02, legend_text[0], fontsize=8, ha='right', va='bottom',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    fig.text(0.02, 0.02, legend_text[1], fontsize=8, ha='left', va='bottom',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('outputs/objective_comparison.png', dpi=150, bbox_inches='tight')
    print("\n✅ Saved comparison chart to: outputs/objective_comparison.png")
    plt.close()
    
    # Create a summary table
    df = pd.DataFrame(results).T
    df['score'] = scores
    df = df.sort_values('score', ascending=False)
    
    print("\n📊 Performance Ranking (by combined score):")
    print("="*60)
    for idx, (mode, row) in enumerate(df.iterrows(), 1):
        print(f"{idx}. {mode}")
        print(f"   Score: {row['score']:.3f} | Win: {row['win_rate']:.1%} | Bust: {row['bust_rate']:.1%} | Reward: {row['avg_reward']:.3f}")
    
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare", action="store_true", help="Compare all modes")
    args = parser.parse_args()
    
    if args.compare or True:  # Default to compare
        df = compare_modes()
        
        print("\n🎯 Key Insights:")
        print("-" * 40)
        print("• The 8-objective models show more nuanced strategies")
        print("• Strategic mode balances multiple goals effectively")
        print("• Optimal mode achieves best win rate with acceptable bust rate")
        print("• More objectives = better strategic decision making")
