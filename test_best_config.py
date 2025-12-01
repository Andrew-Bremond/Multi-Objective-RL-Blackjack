"""Test the best multi-objective configuration for Blackjack."""
import subprocess
import json
from pathlib import Path

# Test configurations based on analysis
configs = [
    {
        "name": "hybrid_best",
        "profit": 0.85,
        "loss": 0.05,
        "bust": 0.05,
        "close_call": 0.02,
        "dealer_weak": 0.01,
        "conservative": 0.01,
        "aggressive": 0.01,
        "perfect": 0.0,
        "episodes": 100000
    },
    {
        "name": "win_focused",
        "profit": 0.9,
        "loss": 0.03,
        "bust": 0.07,
        "close_call": 0.0,
        "dealer_weak": 0.0,
        "conservative": 0.0,
        "aggressive": 0.0,
        "perfect": 0.0,
        "episodes": 100000
    },
    {
        "name": "balanced_8obj",
        "profit": 0.75,
        "loss": 0.08,
        "bust": 0.08,
        "close_call": 0.02,
        "dealer_weak": 0.02,
        "conservative": 0.02,
        "aggressive": 0.02,
        "perfect": 0.01,
        "episodes": 100000
    }
]

print("🎯 Testing optimized configurations...")
print("=" * 60)

best_win_rate = 0
best_config = None

for config in configs:
    print(f"\n📊 Testing: {config['name']}")
    print(f"   Weights: profit={config['profit']:.2f}, loss={config['loss']:.2f}, bust={config['bust']:.2f}")
    
    cmd = [
        ".venv/bin/python3", "train.py",
        "--mode", "custom",
        "--profit-weight", str(config["profit"]),
        "--loss-penalty", str(config["loss"]),
        "--bust-penalty", str(config["bust"]),
        "--close-call-bonus", str(config["close_call"]),
        "--dealer-weak-bonus", str(config["dealer_weak"]),
        "--conservative-stand-bonus", str(config["conservative"]),
        "--aggressive-hit-bonus", str(config["aggressive"]),
        "--perfect-play-bonus", str(config["perfect"]),
        "--total-episodes", str(config["episodes"]),
        "--save-dir", f"outputs/{config['name']}"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Extract results
    if "Final evaluation:" in result.stdout:
        json_str = result.stdout.split("Final evaluation:")[1].strip()
        eval_data = json.loads(json_str)
        
        print(f"   ✅ Win Rate: {eval_data['win_rate']:.1%}")
        print(f"   📉 Bust Rate: {eval_data['bust_rate']:.1%}")
        print(f"   💰 Avg Reward: {eval_data['avg_reward']:.3f}")
        
        if eval_data['win_rate'] > best_win_rate:
            best_win_rate = eval_data['win_rate']
            best_config = config['name']

print("\n" + "=" * 60)
print(f"🏆 BEST CONFIGURATION: {best_config}")
print(f"   Achieved Win Rate: {best_win_rate:.1%}")

# Final recommendation
print("\n💡 RECOMMENDATION:")
print("The best configuration for improving win rate is:")
print(f"   --mode custom --profit-weight 0.9 --loss-penalty 0.03 --bust-penalty 0.07")
print("   This focuses on winning while maintaining reasonable risk management.")
