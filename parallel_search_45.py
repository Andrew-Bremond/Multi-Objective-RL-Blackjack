#!/usr/bin/env python3
"""Parallel hyperparameter search to achieve 45% win rate."""

import subprocess
import json
import concurrent.futures
import time
from pathlib import Path
from typing import Dict, List, Tuple

def train_config(config_data: Tuple[int, Dict]) -> Dict:
    """Train a single configuration."""
    idx, config = config_data
    
    save_dir = f"outputs/parallel/config_{idx}"
    
    cmd = [
        ".venv/bin/python3", "train.py",
        "--mode", "custom",
        "--profit-weight", str(config['profit']),
        "--loss-penalty", str(config['loss']),
        "--bust-penalty", str(config['bust']),
        "--total-episodes", str(config.get('episodes', 50000)),
        "--learning-rate", str(config.get('lr', 4e-4)),
        "--entropy-coef", str(config.get('entropy', 0.04)),
        "--update-epochs", str(config.get('epochs', 10)),
        "--save-dir", save_dir
    ]
    
    # Add bonus objectives if present
    if config.get('close_call', 0) > 0:
        cmd.extend(["--close-call-bonus", str(config['close_call'])])
    if config.get('dealer_weak', 0) > 0:
        cmd.extend(["--dealer-weak-bonus", str(config['dealer_weak'])])
    if config.get('aggressive', 0) > 0:
        cmd.extend(["--aggressive-hit-bonus", str(config['aggressive'])])
    
    print(f"🚀 Starting config {idx}: profit={config['profit']:.2f}, loss={config['loss']:.2f}, bust={config['bust']:.2f}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Extract results
    if "Final evaluation:" in result.stdout:
        json_str = result.stdout.split("Final evaluation:")[1].strip()
        eval_data = json.loads(json_str)
        
        return {
            'config_id': idx,
            'config': config,
            'win_rate': eval_data['win_rate'],
            'bust_rate': eval_data['bust_rate'],
            'avg_reward': eval_data['avg_reward']
        }
    
    return {
        'config_id': idx,
        'config': config,
        'win_rate': 0.0,
        'bust_rate': 1.0,
        'avg_reward': -1.0
    }

def generate_config_grid() -> List[Dict]:
    """Generate a comprehensive grid of configurations to test."""
    configs = []
    
    # High-precision grid around promising areas
    for profit in [0.82, 0.85, 0.87, 0.9, 0.92]:
        for loss_ratio in [0.3, 0.4, 0.5, 0.6]:
            total_penalty = 0.1
            loss = total_penalty * loss_ratio
            bust = total_penalty * (1 - loss_ratio)
            
            configs.append({
                'profit': profit,
                'loss': loss,
                'bust': bust,
                'episodes': 50000,
                'lr': 4e-4,
                'entropy': 0.04,
                'epochs': 10
            })
    
    # Configurations with bonus objectives
    bonus_configs = [
        {'profit': 0.85, 'loss': 0.05, 'bust': 0.05, 'close_call': 0.03, 'dealer_weak': 0.02},
        {'profit': 0.88, 'loss': 0.04, 'bust': 0.06, 'aggressive': 0.02},
        {'profit': 0.9, 'loss': 0.03, 'bust': 0.04, 'close_call': 0.02, 'aggressive': 0.01},
        {'profit': 0.83, 'loss': 0.06, 'bust': 0.08, 'dealer_weak': 0.03},
        {'profit': 0.87, 'loss': 0.045, 'bust': 0.055, 'close_call': 0.015, 'dealer_weak': 0.015},
    ]
    
    for bc in bonus_configs:
        bc['episodes'] = 50000
        bc['lr'] = 4e-4
        bc['entropy'] = 0.04
        bc['epochs'] = 12
        configs.append(bc)
    
    # Different hyperparameter settings
    hp_variations = [
        {'lr': 3e-4, 'entropy': 0.05, 'epochs': 10},
        {'lr': 5e-4, 'entropy': 0.03, 'epochs': 12},
        {'lr': 4e-4, 'entropy': 0.06, 'epochs': 8},
        {'lr': 6e-4, 'entropy': 0.04, 'epochs': 15},
    ]
    
    for hp in hp_variations:
        base_config = {
            'profit': 0.88,
            'loss': 0.05,
            'bust': 0.07,
            'episodes': 50000
        }
        base_config.update(hp)
        configs.append(base_config)
    
    return configs

def parallel_search(max_workers=4):
    """Run parallel search for best configuration."""
    print("🚀 PARALLEL HYPERPARAMETER SEARCH FOR 45% WIN RATE")
    print(f"   Running with {max_workers} parallel workers")
    print("=" * 60)
    
    configs = generate_config_grid()
    print(f"📊 Testing {len(configs)} configurations...")
    
    start_time = time.time()
    results = []
    
    # Process in batches
    batch_size = max_workers * 2
    for i in range(0, len(configs), batch_size):
        batch = list(enumerate(configs[i:i+batch_size], start=i))
        
        print(f"\n🔄 Processing batch {i//batch_size + 1}/{(len(configs) + batch_size - 1)//batch_size}")
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            batch_results = list(executor.map(train_config, batch))
            results.extend(batch_results)
        
        # Check if we've achieved target
        for result in batch_results:
            if result['win_rate'] >= 0.45:
                print(f"\n🎉 TARGET ACHIEVED! Win Rate: {result['win_rate']:.1%}")
                print(f"   Config: {result['config']}")
                save_results(results, result)
                return
        
        # Print batch summary
        best_in_batch = max(batch_results, key=lambda x: x['win_rate'])
        print(f"   Best in batch: {best_in_batch['win_rate']:.1%}")
    
    # Find overall best
    best_result = max(results, key=lambda x: x['win_rate'])
    
    elapsed = time.time() - start_time
    print(f"\n⏱️ Search completed in {elapsed:.1f} seconds")
    
    save_results(results, best_result)

def save_results(results: List[Dict], best_result: Dict):
    """Save search results."""
    print("\n" + "=" * 60)
    print("📊 SEARCH RESULTS")
    print("=" * 60)
    
    # Sort by win rate
    sorted_results = sorted(results, key=lambda x: x['win_rate'], reverse=True)
    
    print("\n🏆 TOP 5 CONFIGURATIONS:")
    for i, result in enumerate(sorted_results[:5], 1):
        print(f"\n{i}. Win Rate: {result['win_rate']:.1%}, Bust: {result['bust_rate']:.1%}, Reward: {result['avg_reward']:.3f}")
        print(f"   Config: profit={result['config']['profit']:.2f}, loss={result['config']['loss']:.3f}, bust={result['config']['bust']:.3f}")
        if result['config'].get('close_call', 0) > 0:
            print(f"   Bonuses: close_call={result['config']['close_call']:.3f}", end="")
        if result['config'].get('dealer_weak', 0) > 0:
            print(f", dealer_weak={result['config']['dealer_weak']:.3f}", end="")
        if result['config'].get('aggressive', 0) > 0:
            print(f", aggressive={result['config']['aggressive']:.3f}", end="")
        if any([result['config'].get(k, 0) > 0 for k in ['close_call', 'dealer_weak', 'aggressive']]):
            print()
    
    # Save to file
    Path("outputs/parallel").mkdir(parents=True, exist_ok=True)
    
    with open("outputs/parallel/results.json", "w") as f:
        json.dump({
            'best_win_rate': best_result['win_rate'],
            'best_config': best_result['config'],
            'top_5': sorted_results[:5],
            'total_tested': len(results)
        }, f, indent=2)
    
    print(f"\n💾 Results saved to outputs/parallel/")
    
    if best_result['win_rate'] >= 0.45:
        print(f"\n✅ SUCCESS! Configuration achieving {best_result['win_rate']:.1%} win rate:")
    else:
        print(f"\n⚠️ Best configuration achieved {best_result['win_rate']:.1%} win rate:")
    
    config = best_result['config']
    print(f"\npython train.py --mode custom \\")
    print(f"  --profit-weight {config['profit']} \\")
    print(f"  --loss-penalty {config['loss']} \\")
    print(f"  --bust-penalty {config['bust']} \\")
    if config.get('close_call', 0) > 0:
        print(f"  --close-call-bonus {config['close_call']} \\")
    if config.get('dealer_weak', 0) > 0:
        print(f"  --dealer-weak-bonus {config['dealer_weak']} \\")
    if config.get('aggressive', 0) > 0:
        print(f"  --aggressive-hit-bonus {config['aggressive']} \\")
    print(f"  --learning-rate {config.get('lr', 4e-4)} \\")
    print(f"  --entropy-coef {config.get('entropy', 0.04)} \\")
    print(f"  --update-epochs {config.get('epochs', 10)} \\")
    print(f"  --total-episodes 100000")

if __name__ == "__main__":
    import sys
    max_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    parallel_search(max_workers=max_workers)
