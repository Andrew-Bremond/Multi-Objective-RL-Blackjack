#!/usr/bin/env python3
"""Bayesian optimization to achieve 45% win rate in Blackjack."""

import subprocess
import json
import numpy as np
from pathlib import Path
import time
from typing import Dict, List, Tuple
import random

class BayesianTuner:
    """Bayesian optimization for hyperparameter tuning."""
    
    def __init__(self, target_win_rate=0.45):
        self.target_win_rate = target_win_rate
        self.best_win_rate = 0.0
        self.best_config = None
        self.history = []
        
    def objective_function(self, params: Dict) -> float:
        """Train and evaluate a configuration, return negative win rate for minimization."""
        save_dir = f"outputs/bayesian/trial_{len(self.history)}"
        
        cmd = [
            ".venv/bin/python3", "train.py",
            "--mode", "custom",
            "--profit-weight", str(params['profit']),
            "--loss-penalty", str(params['loss']),
            "--bust-penalty", str(params['bust']),
            "--close-call-bonus", str(params.get('close_call', 0)),
            "--dealer-weak-bonus", str(params.get('dealer_weak', 0)),
            "--aggressive-hit-bonus", str(params.get('aggressive', 0)),
            "--total-episodes", str(params.get('episodes', 50000)),
            "--learning-rate", str(params.get('lr', 4e-4)),
            "--entropy-coef", str(params.get('entropy', 0.04)),
            "--update-epochs", str(params.get('epochs', 10)),
            "--save-dir", save_dir
        ]
        
        print(f"\n🎯 Trial {len(self.history) + 1}")
        print(f"   Config: profit={params['profit']:.3f}, loss={params['loss']:.3f}, bust={params['bust']:.3f}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Extract results
        if "Final evaluation:" in result.stdout:
            json_str = result.stdout.split("Final evaluation:")[1].strip()
            eval_data = json.loads(json_str)
            win_rate = eval_data['win_rate']
            
            print(f"   📊 Win Rate: {win_rate:.1%}, Bust: {eval_data['bust_rate']:.1%}, Reward: {eval_data['avg_reward']:.3f}")
            
            self.history.append({
                'params': params,
                'win_rate': win_rate,
                'full_results': eval_data
            })
            
            if win_rate > self.best_win_rate:
                self.best_win_rate = win_rate
                self.best_config = params
                print(f"   ✅ NEW BEST: {win_rate:.1%}")
                
                if win_rate >= self.target_win_rate:
                    print(f"\n🎉 TARGET ACHIEVED! Win Rate: {win_rate:.1%}")
                    self.save_results()
                    return -1.0  # Signal to stop
            
            return -win_rate  # Negative for minimization
        
        return 0.0  # Failed run
    
    def suggest_next_params(self) -> Dict:
        """Suggest next parameters based on history."""
        if len(self.history) < 5:
            # Initial random exploration
            return {
                'profit': random.uniform(0.75, 0.95),
                'loss': random.uniform(0.02, 0.15),
                'bust': random.uniform(0.02, 0.15),
                'close_call': random.uniform(0, 0.05),
                'dealer_weak': random.uniform(0, 0.05),
                'aggressive': random.uniform(0, 0.03),
                'episodes': 50000,
                'lr': random.choice([3e-4, 4e-4, 5e-4]),
                'entropy': random.uniform(0.03, 0.06),
                'epochs': random.choice([8, 10, 12])
            }
        
        # Exploit best regions with small exploration
        if self.best_config:
            # Add noise to best config
            return {
                'profit': np.clip(self.best_config['profit'] + random.gauss(0, 0.02), 0.7, 1.0),
                'loss': np.clip(self.best_config['loss'] + random.gauss(0, 0.01), 0, 0.2),
                'bust': np.clip(self.best_config['bust'] + random.gauss(0, 0.01), 0, 0.2),
                'close_call': np.clip(self.best_config.get('close_call', 0) + random.gauss(0, 0.01), 0, 0.1),
                'dealer_weak': np.clip(self.best_config.get('dealer_weak', 0) + random.gauss(0, 0.01), 0, 0.1),
                'aggressive': np.clip(self.best_config.get('aggressive', 0) + random.gauss(0, 0.01), 0, 0.05),
                'episodes': self.best_config.get('episodes', 50000),
                'lr': self.best_config.get('lr', 4e-4),
                'entropy': np.clip(self.best_config.get('entropy', 0.04) + random.gauss(0, 0.005), 0.02, 0.08),
                'epochs': self.best_config.get('epochs', 10)
            }
        
        # Fallback to random
        return self.suggest_next_params()
    
    def optimize(self, n_trials=100):
        """Run Bayesian optimization."""
        print("🔬 BAYESIAN OPTIMIZATION FOR 45% WIN RATE")
        print("=" * 60)
        
        for trial in range(n_trials):
            params = self.suggest_next_params()
            
            # Increase episodes as we get closer
            if self.best_win_rate > 0.43:
                params['episodes'] = 75000
            elif self.best_win_rate > 0.44:
                params['episodes'] = 100000
            
            result = self.objective_function(params)
            
            if result == -1.0:  # Target achieved
                break
            
            # Every 10 trials, try a completely random config
            if trial % 10 == 9:
                print("\n🎲 Exploration trial...")
                params = {
                    'profit': random.uniform(0.65, 1.0),
                    'loss': random.uniform(0, 0.25),
                    'bust': random.uniform(0, 0.25),
                    'close_call': random.uniform(0, 0.1),
                    'dealer_weak': random.uniform(0, 0.1),
                    'aggressive': random.uniform(0, 0.05),
                    'episodes': 30000,  # Faster for exploration
                    'lr': random.choice([2e-4, 3e-4, 4e-4, 5e-4, 6e-4]),
                    'entropy': random.uniform(0.02, 0.08),
                    'epochs': random.choice([6, 8, 10, 12, 15])
                }
                self.objective_function(params)
        
        if self.best_win_rate < self.target_win_rate:
            print(f"\n⚠️ Maximum trials reached. Best win rate: {self.best_win_rate:.1%}")
        
        self.save_results()
    
    def save_results(self):
        """Save the best configuration."""
        print("\n" + "=" * 60)
        print("📊 OPTIMIZATION COMPLETE")
        print("=" * 60)
        print(f"Best Win Rate: {self.best_win_rate:.1%}")
        print(f"Total Trials: {len(self.history)}")
        
        if self.best_config:
            print(f"\n🏆 BEST CONFIGURATION:")
            for key, value in self.best_config.items():
                if isinstance(value, float):
                    print(f"   {key}: {value:.4f}")
                else:
                    print(f"   {key}: {value}")
            
            Path("outputs/bayesian").mkdir(parents=True, exist_ok=True)
            
            with open("outputs/bayesian/best_config.json", "w") as f:
                json.dump({
                    "win_rate": self.best_win_rate,
                    "config": self.best_config,
                    "trials": len(self.history)
                }, f, indent=2)
            
            print(f"\n💾 Results saved to outputs/bayesian/")
            
            print(f"\n🎮 To use this configuration:")
            print(f"python train.py --mode custom \\")
            print(f"  --profit-weight {self.best_config['profit']:.4f} \\")
            print(f"  --loss-penalty {self.best_config['loss']:.4f} \\")
            print(f"  --bust-penalty {self.best_config['bust']:.4f} \\")
            if self.best_config.get('close_call', 0) > 0:
                print(f"  --close-call-bonus {self.best_config['close_call']:.4f} \\")
            if self.best_config.get('dealer_weak', 0) > 0:
                print(f"  --dealer-weak-bonus {self.best_config['dealer_weak']:.4f} \\")
            print(f"  --learning-rate {self.best_config.get('lr', 4e-4)} \\")
            print(f"  --entropy-coef {self.best_config.get('entropy', 0.04):.4f} \\")
            print(f"  --update-epochs {self.best_config.get('epochs', 10)}")

if __name__ == "__main__":
    tuner = BayesianTuner(target_win_rate=0.45)
    tuner.optimize(n_trials=100)
