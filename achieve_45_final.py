#!/usr/bin/env python3
"""Final push to achieve 45% win rate - intensive search."""

import subprocess
import json
import numpy as np
from pathlib import Path
import time

def train_config(config_name, params):
    """Train a specific configuration."""
    cmd = [
        ".venv/bin/python3", "train.py",
        "--mode", "custom",
        "--profit-weight", str(params['p']),
        "--loss-penalty", str(params['l']),
        "--bust-penalty", str(params['b']),
        "--total-episodes", str(params.get('ep', 75000)),
        "--learning-rate", str(params.get('lr', 0.0004)),
        "--entropy-coef", str(params.get('ent', 0.04)),
        "--update-epochs", str(params.get('epochs', 12)),
        "--save-dir", f"outputs/final_45/{config_name}"
    ]
    
    # Add optional bonuses
    if params.get('cc', 0) > 0:
        cmd.extend(["--close-call-bonus", str(params['cc'])])
    if params.get('dw', 0) > 0:
        cmd.extend(["--dealer-weak-bonus", str(params['dw'])])
    if params.get('ah', 0) > 0:
        cmd.extend(["--aggressive-hit-bonus", str(params['ah'])])
    
    print(f"\n🎯 Testing {config_name}:")
    print(f"   Weights: p={params['p']:.3f}, l={params['l']:.3f}, b={params['b']:.3f}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if "Final evaluation:" in result.stdout:
        json_str = result.stdout.split("Final evaluation:")[1].strip()
        eval_data = json.loads(json_str)
        return eval_data['win_rate'], eval_data
    
    return 0.0, None

print("=" * 60)
print("🎯 FINAL PUSH TO ACHIEVE 45% WIN RATE")
print("=" * 60)

# Based on our best results so far (44.6%), let's fine-tune around that region
best_configs = [
    # Variations of the 44.6% config
    ("base_446", {'p': 0.90, 'l': 0.03, 'b': 0.04, 'cc': 0.02, 'ah': 0.01, 'ep': 75000}),
    ("var1", {'p': 0.91, 'l': 0.028, 'b': 0.038, 'cc': 0.018, 'ah': 0.008, 'ep': 75000}),
    ("var2", {'p': 0.89, 'l': 0.032, 'b': 0.042, 'cc': 0.022, 'ah': 0.012, 'ep': 75000}),
    ("var3", {'p': 0.90, 'l': 0.025, 'b': 0.045, 'cc': 0.025, 'ah': 0.005, 'ep': 75000}),
    ("var4", {'p': 0.905, 'l': 0.027, 'b': 0.037, 'cc': 0.019, 'ah': 0.009, 'ep': 75000}),
    
    # Different learning rates
    ("lr1", {'p': 0.90, 'l': 0.03, 'b': 0.04, 'cc': 0.02, 'ah': 0.01, 'lr': 0.0003, 'ep': 75000}),
    ("lr2", {'p': 0.90, 'l': 0.03, 'b': 0.04, 'cc': 0.02, 'ah': 0.01, 'lr': 0.0005, 'ep': 75000}),
    
    # Different entropy
    ("ent1", {'p': 0.90, 'l': 0.03, 'b': 0.04, 'cc': 0.02, 'ah': 0.01, 'ent': 0.035, 'ep': 75000}),
    ("ent2", {'p': 0.90, 'l': 0.03, 'b': 0.04, 'cc': 0.02, 'ah': 0.01, 'ent': 0.045, 'ep': 75000}),
    
    # More epochs
    ("ep1", {'p': 0.90, 'l': 0.03, 'b': 0.04, 'cc': 0.02, 'ah': 0.01, 'epochs': 15, 'ep': 75000}),
    ("ep2", {'p': 0.90, 'l': 0.03, 'b': 0.04, 'cc': 0.02, 'ah': 0.01, 'epochs': 18, 'ep': 75000}),
    
    # Slightly different penalty ratios
    ("ratio1", {'p': 0.88, 'l': 0.04, 'b': 0.05, 'cc': 0.02, 'ah': 0.01, 'ep': 75000}),
    ("ratio2", {'p': 0.92, 'l': 0.02, 'b': 0.03, 'cc': 0.02, 'ah': 0.01, 'ep': 75000}),
    ("ratio3", {'p': 0.895, 'l': 0.035, 'b': 0.035, 'cc': 0.02, 'ah': 0.015, 'ep': 75000}),
    
    # Add dealer weak bonus
    ("dw1", {'p': 0.90, 'l': 0.03, 'b': 0.04, 'cc': 0.015, 'dw': 0.015, 'ah': 0.01, 'ep': 75000}),
    ("dw2", {'p': 0.89, 'l': 0.03, 'b': 0.04, 'cc': 0.02, 'dw': 0.02, 'ah': 0.0, 'ep': 75000}),
]

best_win_rate = 0.0
best_config = None
results = []

for config_name, params in best_configs:
    win_rate, eval_data = train_config(config_name, params)
    
    if eval_data:
        print(f"   ✅ Win: {win_rate:.1%}, Bust: {eval_data['bust_rate']:.1%}, Reward: {eval_data['avg_reward']:.3f}")
        results.append((config_name, params, win_rate, eval_data))
        
        if win_rate > best_win_rate:
            best_win_rate = win_rate
            best_config = (config_name, params, eval_data)
            print(f"   🏆 NEW BEST: {win_rate:.1%}")
        
        if win_rate >= 0.45:
            print(f"\n🎉 SUCCESS! ACHIEVED {win_rate:.1%} WIN RATE!")
            break

print("\n" + "=" * 60)
print("📊 FINAL RESULTS")
print("=" * 60)

# Sort results by win rate
results.sort(key=lambda x: x[2], reverse=True)

print("\n🏆 TOP CONFIGURATIONS:")
for i, (name, params, win_rate, data) in enumerate(results[:5], 1):
    print(f"\n{i}. {name}: Win={win_rate:.1%}, Bust={data['bust_rate']:.1%}, Reward={data['avg_reward']:.3f}")
    print(f"   Config: p={params['p']:.3f}, l={params['l']:.3f}, b={params['b']:.3f}")

if best_win_rate >= 0.45:
    print(f"\n✅ TARGET ACHIEVED!")
    name, params, data = best_config
    print(f"Configuration '{name}' achieved {best_win_rate:.1%} win rate")
    
    # Save the winning configuration
    Path("outputs/final_45").mkdir(parents=True, exist_ok=True)
    with open("outputs/final_45/WINNING_CONFIG.json", "w") as f:
        json.dump({
            "win_rate": best_win_rate,
            "config_name": name,
            "params": params,
            "full_results": data
        }, f, indent=2)
    
    print(f"\n🎮 Use this configuration:")
    print(f"python train.py --mode custom \\")
    print(f"  --profit-weight {params['p']} \\")
    print(f"  --loss-penalty {params['l']} \\")
    print(f"  --bust-penalty {params['b']} \\")
    if params.get('cc', 0) > 0:
        print(f"  --close-call-bonus {params['cc']} \\")
    if params.get('dw', 0) > 0:
        print(f"  --dealer-weak-bonus {params['dw']} \\")
    if params.get('ah', 0) > 0:
        print(f"  --aggressive-hit-bonus {params['ah']} \\")
    print(f"  --learning-rate {params.get('lr', 0.0004)} \\")
    print(f"  --entropy-coef {params.get('ent', 0.04)} \\")
    print(f"  --update-epochs {params.get('epochs', 12)} \\")
    print(f"  --total-episodes 150000")
else:
    print(f"\n⚠️ Best achieved: {best_win_rate:.1%} (target was 45%)")
    print("Next steps:")
    print("1. Try with more episodes (150K-200K)")
    print("2. Adjust network architecture")
    print("3. Fine-tune learning rate schedule")
    print("4. Add more sophisticated objectives")
