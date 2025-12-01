"""Compare training results before and after hyperparameter improvements."""
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def plot_comparison():
    """Create comparison plots for baseline and risk-averse modes."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training Performance: Hyperparameter Improvements', fontsize=16, fontweight='bold')
    
    # Load baseline improved data
    baseline_path = Path('outputs/baseline_improved/metrics.csv')
    if baseline_path.exists():
        df_baseline = pd.read_csv(baseline_path)
        
        # Win rate
        axes[0, 0].plot(df_baseline['episodes'], df_baseline['win_rate'], 'b-', linewidth=2, label='Win Rate')
        axes[0, 0].plot(df_baseline['episodes'], df_baseline['loss_rate'], 'r-', linewidth=2, label='Loss Rate')
        axes[0, 0].plot(df_baseline['episodes'], df_baseline['bust_rate'], 'orange', linewidth=2, label='Bust Rate')
        axes[0, 0].axhline(y=0.346, color='b', linestyle='--', alpha=0.5, label='Old Win Rate')
        axes[0, 0].axhline(y=0.594, color='r', linestyle='--', alpha=0.5, label='Old Loss Rate')
        axes[0, 0].set_xlabel('Episodes', fontsize=11)
        axes[0, 0].set_ylabel('Rate', fontsize=11)
        axes[0, 0].set_title('Baseline: Win/Loss/Bust Rates', fontsize=12, fontweight='bold')
        axes[0, 0].legend(fontsize=9)
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim(0, 1)
        
        # Average reward
        axes[0, 1].plot(df_baseline['episodes'], df_baseline['avg_reward'], 'g-', linewidth=2, label='Average Reward')
        axes[0, 1].axhline(y=-0.248, color='r', linestyle='--', alpha=0.5, label='Old Avg Reward')
        axes[0, 1].axhline(y=0, color='gray', linestyle='-', alpha=0.3, label='Break-even')
        axes[0, 1].set_xlabel('Episodes', fontsize=11)
        axes[0, 1].set_ylabel('Average Reward', fontsize=11)
        axes[0, 1].set_title('Baseline: Average Reward Over Time', fontsize=12, fontweight='bold')
        axes[0, 1].legend(fontsize=9)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Add final performance text
        final_win = df_baseline['win_rate'].iloc[-1]
        final_reward = df_baseline['avg_reward'].iloc[-1]
        axes[0, 0].text(0.98, 0.02, f'Final Win Rate: {final_win:.1%}', 
                       transform=axes[0, 0].transAxes, ha='right', va='bottom',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=9)
        axes[0, 1].text(0.98, 0.02, f'Final Reward: {final_reward:.3f}', 
                       transform=axes[0, 1].transAxes, ha='right', va='bottom',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=9)
    
    # Load risk-averse improved data
    risk_path = Path('outputs/risk_averse_improved/metrics.csv')
    if risk_path.exists():
        df_risk = pd.read_csv(risk_path)
        
        # Win rate
        axes[1, 0].plot(df_risk['episodes'], df_risk['win_rate'], 'b-', linewidth=2, label='Win Rate')
        axes[1, 0].plot(df_risk['episodes'], df_risk['loss_rate'], 'r-', linewidth=2, label='Loss Rate')
        axes[1, 0].plot(df_risk['episodes'], df_risk['bust_rate'], 'orange', linewidth=2, label='Bust Rate')
        axes[1, 0].axhline(y=0.356, color='b', linestyle='--', alpha=0.5, label='Old Win Rate')
        axes[1, 0].axhline(y=0.582, color='r', linestyle='--', alpha=0.5, label='Old Loss Rate')
        axes[1, 0].axhline(y=0.06, color='orange', linestyle='--', alpha=0.5, label='Old Bust Rate')
        axes[1, 0].set_xlabel('Episodes', fontsize=11)
        axes[1, 0].set_ylabel('Rate', fontsize=11)
        axes[1, 0].set_title('Risk-Averse: Win/Loss/Bust Rates', fontsize=12, fontweight='bold')
        axes[1, 0].legend(fontsize=9)
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim(0, 1)
        
        # Average reward
        axes[1, 1].plot(df_risk['episodes'], df_risk['avg_reward'], 'g-', linewidth=2, label='Average Reward')
        axes[1, 1].axhline(y=-0.226, color='r', linestyle='--', alpha=0.5, label='Old Avg Reward')
        axes[1, 1].axhline(y=0, color='gray', linestyle='-', alpha=0.3, label='Break-even')
        axes[1, 1].set_xlabel('Episodes', fontsize=11)
        axes[1, 1].set_ylabel('Average Reward', fontsize=11)
        axes[1, 1].set_title('Risk-Averse: Average Reward Over Time', fontsize=12, fontweight='bold')
        axes[1, 1].legend(fontsize=9)
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add final performance text
        final_win = df_risk['win_rate'].iloc[-1]
        final_reward = df_risk['avg_reward'].iloc[-1]
        final_bust = df_risk['bust_rate'].iloc[-1]
        axes[1, 0].text(0.98, 0.02, f'Final Bust Rate: {final_bust:.1%}', 
                       transform=axes[1, 0].transAxes, ha='right', va='bottom',
                       bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5), fontsize=9)
        axes[1, 1].text(0.98, 0.02, f'Final Reward: {final_reward:.3f}', 
                       transform=axes[1, 1].transAxes, ha='right', va='bottom',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=9)
    
    plt.tight_layout()
    plt.savefig('outputs/performance_comparison.png', dpi=150, bbox_inches='tight')
    print("✅ Saved performance comparison to: outputs/performance_comparison.png")
    plt.close()
    
    # Create a bar chart comparison
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    metrics = ['Win Rate', 'Loss Rate', 'Bust Rate', 'Avg Reward']
    
    # Old values
    old_baseline = [0.346, 0.594, 0.074, -0.248]
    old_risk = [0.356, 0.582, 0.060, -0.226]
    
    # New values
    new_baseline = [0.419, 0.491, 0.172, -0.072]
    new_risk = [0.407, 0.500, 0.030, -0.093]
    
    x = range(len(metrics))
    width = 0.2
    
    ax.bar([i - 1.5*width for i in x], old_baseline, width, label='Old Baseline', color='lightcoral', alpha=0.7)
    ax.bar([i - 0.5*width for i in x], new_baseline, width, label='New Baseline', color='darkred', alpha=0.9)
    ax.bar([i + 0.5*width for i in x], old_risk, width, label='Old Risk-Averse', color='lightblue', alpha=0.7)
    ax.bar([i + 1.5*width for i in x], new_risk, width, label='New Risk-Averse', color='darkblue', alpha=0.9)
    
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('Performance Comparison: Before vs After Improvements', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    
    plt.tight_layout()
    plt.savefig('outputs/metrics_bar_comparison.png', dpi=150, bbox_inches='tight')
    print("✅ Saved bar comparison to: outputs/metrics_bar_comparison.png")
    plt.close()

if __name__ == '__main__':
    plot_comparison()
    print("\n📊 Comparison visualizations created successfully!")

