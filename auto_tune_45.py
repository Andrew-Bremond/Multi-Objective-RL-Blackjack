#!/usr/bin/env python3
"""Automated hyperparameter tuning to achieve 45% win rate in Blackjack."""

import subprocess
import json
import numpy as np
from pathlib import Path
import time
from typing import Dict, List, Tuple
import random
from dataclasses import dataclass
import pickle

@dataclass
class Configuration:
    """A configuration with all objective weights."""
    profit: float
    loss_penalty: float
    bust_penalty: float
    close_call_bonus: float = 0.0
    dealer_weak_bonus: float = 0.0
    conservative_stand_bonus: float = 0.0
    aggressive_hit_bonus: float = 0.0
    perfect_play_bonus: float = 0.0
    blackjack_bonus: float = 0.0
    push_penalty: float = 0.0
    early_stand_penalty: float = 0.0
    dealer_bust_bonus: float = 0.0
    
    def to_dict(self):
        return self.__dict__
    
    def to_cmd_args(self):
        """Convert to command line arguments."""
        return [
            "--profit-weight", str(self.profit),
            "--loss-penalty", str(self.loss_penalty),
            "--bust-penalty", str(self.bust_penalty),
            "--close-call-bonus", str(self.close_call_bonus),
            "--dealer-weak-bonus", str(self.dealer_weak_bonus),
            "--conservative-stand-bonus", str(self.conservative_stand_bonus),
            "--aggressive-hit-bonus", str(self.aggressive_hit_bonus),
            "--perfect-play-bonus", str(self.perfect_play_bonus),
        ]

class HyperparameterTuner:
    """Automated tuner to reach 45% win rate."""
    
    def __init__(self, target_win_rate=0.45, max_iterations=100):
        self.target_win_rate = target_win_rate
        self.max_iterations = max_iterations
        self.best_config = None
        self.best_win_rate = 0.0
        self.history = []
        self.iteration = 0
        
    def train_and_evaluate(self, config: Configuration, episodes: int = 50000) -> Dict:
        """Train a model with given configuration and return results."""
        save_dir = f"outputs/auto_tune/iter_{self.iteration}"
        
        cmd = [
            ".venv/bin/python3", "train.py",
            "--mode", "custom",
            *config.to_cmd_args(),
            "--total-episodes", str(episodes),
            "--save-dir", save_dir,
            "--learning-rate", "4e-4",
            "--entropy-coef", "0.04",
            "--update-epochs", "12"
        ]
        
        print(f"   Training with: profit={config.profit:.3f}, loss={config.loss_penalty:.3f}, bust={config.bust_penalty:.3f}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Extract results
        if "Final evaluation:" in result.stdout:
            json_str = result.stdout.split("Final evaluation:")[1].strip()
            eval_data = json.loads(json_str)
            return eval_data
        return {"win_rate": 0.0, "loss_rate": 1.0, "bust_rate": 0.5, "avg_reward": -1.0}
    
    def generate_initial_configs(self) -> List[Configuration]:
        """Generate diverse initial configurations."""
        configs = []
        
        # Known good starting points
        configs.append(Configuration(0.85, 0.075, 0.075))  # Balanced
        configs.append(Configuration(0.9, 0.05, 0.05))     # Win-focused
        configs.append(Configuration(0.8, 0.1, 0.1))       # Conservative
        configs.append(Configuration(0.95, 0.025, 0.025))  # Aggressive
        configs.append(Configuration(0.7, 0.15, 0.15))     # Risk-averse
        
        # Add some with bonus objectives
        configs.append(Configuration(0.8, 0.05, 0.05, 
                                    close_call_bonus=0.05, 
                                    dealer_weak_bonus=0.05))
        configs.append(Configuration(0.85, 0.03, 0.07,
                                    aggressive_hit_bonus=0.03,
                                    perfect_play_bonus=0.02))
        
        return configs
    
    def mutate_config(self, config: Configuration, mutation_rate: float = 0.1) -> Configuration:
        """Create a mutated version of a configuration."""
        new_config = Configuration(**config.to_dict())
        
        # Mutate main weights
        if random.random() < mutation_rate:
            new_config.profit = np.clip(config.profit + random.uniform(-0.05, 0.05), 0.5, 1.0)
        if random.random() < mutation_rate:
            new_config.loss_penalty = np.clip(config.loss_penalty + random.uniform(-0.02, 0.02), 0.0, 0.3)
        if random.random() < mutation_rate:
            new_config.bust_penalty = np.clip(config.bust_penalty + random.uniform(-0.02, 0.02), 0.0, 0.3)
        
        # Occasionally add bonus objectives
        if random.random() < mutation_rate / 2:
            new_config.close_call_bonus = random.uniform(0, 0.05)
        if random.random() < mutation_rate / 2:
            new_config.dealer_weak_bonus = random.uniform(0, 0.05)
        if random.random() < mutation_rate / 2:
            new_config.aggressive_hit_bonus = random.uniform(0, 0.03)
            
        return new_config
    
    def grid_search_phase(self):
        """Initial grid search to find promising regions."""
        print("\n🔍 PHASE 1: Grid Search")
        print("=" * 60)
        
        profit_values = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        penalty_values = [0.02, 0.05, 0.08, 0.1, 0.15]
        
        for profit in profit_values:
            for penalty in penalty_values:
                self.iteration += 1
                config = Configuration(
                    profit=profit,
                    loss_penalty=penalty * 0.4,  # Less loss penalty
                    bust_penalty=penalty * 0.6   # More bust penalty
                )
                
                print(f"\n🎯 Iteration {self.iteration}/{self.max_iterations}")
                results = self.train_and_evaluate(config, episodes=30000)  # Faster initial search
                
                self.history.append({
                    "iteration": self.iteration,
                    "config": config.to_dict(),
                    "results": results
                })
                
                if results["win_rate"] > self.best_win_rate:
                    self.best_win_rate = results["win_rate"]
                    self.best_config = config
                    print(f"   ✅ NEW BEST: Win Rate = {self.best_win_rate:.1%}")
                
                print(f"   📊 Results: Win={results['win_rate']:.1%}, Bust={results['bust_rate']:.1%}, Reward={results['avg_reward']:.3f}")
                
                if self.best_win_rate >= self.target_win_rate:
                    print(f"\n🎉 TARGET ACHIEVED! Win Rate: {self.best_win_rate:.1%}")
                    return True
                    
                if self.iteration >= self.max_iterations:
                    return False
        
        return False
    
    def evolutionary_phase(self):
        """Use evolutionary algorithm to refine best configurations."""
        print("\n🧬 PHASE 2: Evolutionary Optimization")
        print("=" * 60)
        
        # Start with top configurations from history
        sorted_history = sorted(self.history, key=lambda x: x["results"]["win_rate"], reverse=True)
        population = [Configuration(**h["config"]) for h in sorted_history[:5]]
        
        generation = 0
        while self.iteration < self.max_iterations:
            generation += 1
            print(f"\n📈 Generation {generation}")
            
            # Create offspring through mutation
            offspring = []
            for parent in population:
                for _ in range(2):  # 2 offspring per parent
                    offspring.append(self.mutate_config(parent, mutation_rate=0.15))
            
            # Evaluate offspring
            for config in offspring:
                self.iteration += 1
                print(f"\n🎯 Iteration {self.iteration}/{self.max_iterations}")
                
                # Increase episodes as we get closer to target
                episodes = 50000 if self.best_win_rate > 0.42 else 30000
                results = self.train_and_evaluate(config, episodes=episodes)
                
                self.history.append({
                    "iteration": self.iteration,
                    "config": config.to_dict(),
                    "results": results
                })
                
                if results["win_rate"] > self.best_win_rate:
                    self.best_win_rate = results["win_rate"]
                    self.best_config = config
                    print(f"   ✅ NEW BEST: Win Rate = {self.best_win_rate:.1%}")
                
                print(f"   📊 Results: Win={results['win_rate']:.1%}, Bust={results['bust_rate']:.1%}, Reward={results['avg_reward']:.3f}")
                
                if self.best_win_rate >= self.target_win_rate:
                    print(f"\n🎉 TARGET ACHIEVED! Win Rate: {self.best_win_rate:.1%}")
                    return True
                
                if self.iteration >= self.max_iterations:
                    return False
            
            # Select next generation (top performers)
            all_configs = population + offspring
            all_results = []
            for c in all_configs:
                # Find matching result
                for h in self.history:
                    if h["config"] == c.to_dict():
                        all_results.append((c, h["results"]["win_rate"]))
                        break
            
            all_results.sort(key=lambda x: x[1], reverse=True)
            population = [c for c, _ in all_results[:5]]
    
    def fine_tune_phase(self):
        """Fine-tune the best configuration with small adjustments."""
        print("\n🎯 PHASE 3: Fine-Tuning Best Configuration")
        print("=" * 60)
        
        if not self.best_config:
            return False
        
        # Try small variations around best config
        variations = [
            (0.01, 0, 0),
            (-0.01, 0, 0),
            (0, 0.01, 0),
            (0, -0.01, 0),
            (0, 0, 0.01),
            (0, 0, -0.01),
            (0.02, -0.01, -0.01),
            (-0.02, 0.01, 0.01),
        ]
        
        for dp, dl, db in variations:
            self.iteration += 1
            config = Configuration(
                profit=np.clip(self.best_config.profit + dp, 0.5, 1.0),
                loss_penalty=np.clip(self.best_config.loss_penalty + dl, 0.0, 0.3),
                bust_penalty=np.clip(self.best_config.bust_penalty + db, 0.0, 0.3),
                close_call_bonus=self.best_config.close_call_bonus,
                dealer_weak_bonus=self.best_config.dealer_weak_bonus,
                aggressive_hit_bonus=self.best_config.aggressive_hit_bonus
            )
            
            print(f"\n🎯 Iteration {self.iteration}/{self.max_iterations}")
            results = self.train_and_evaluate(config, episodes=75000)  # More episodes for fine-tuning
            
            self.history.append({
                "iteration": self.iteration,
                "config": config.to_dict(),
                "results": results
            })
            
            if results["win_rate"] > self.best_win_rate:
                self.best_win_rate = results["win_rate"]
                self.best_config = config
                print(f"   ✅ NEW BEST: Win Rate = {self.best_win_rate:.1%}")
            
            print(f"   📊 Results: Win={results['win_rate']:.1%}, Bust={results['bust_rate']:.1%}, Reward={results['avg_reward']:.3f}")
            
            if self.best_win_rate >= self.target_win_rate:
                print(f"\n🎉 TARGET ACHIEVED! Win Rate: {self.best_win_rate:.1%}")
                return True
            
            if self.iteration >= self.max_iterations:
                return False
        
        return False
    
    def run(self):
        """Run the complete tuning process."""
        print("🚀 AUTOMATED HYPERPARAMETER TUNING")
        print(f"   Target: {self.target_win_rate:.1%} win rate")
        print(f"   Max iterations: {self.max_iterations}")
        print("=" * 60)
        
        start_time = time.time()
        
        # Phase 1: Grid Search
        if self.grid_search_phase():
            self.save_results()
            return
        
        # Phase 2: Evolutionary Algorithm
        if self.evolutionary_phase():
            self.save_results()
            return
        
        # Phase 3: Fine-tuning
        if self.fine_tune_phase():
            self.save_results()
            return
        
        # If we didn't reach target
        print(f"\n⚠️ Maximum iterations reached. Best win rate: {self.best_win_rate:.1%}")
        self.save_results()
    
    def save_results(self):
        """Save the best configuration and history."""
        elapsed_time = time.time()
        
        print("\n" + "=" * 60)
        print("📊 FINAL RESULTS")
        print("=" * 60)
        print(f"Best Win Rate Achieved: {self.best_win_rate:.1%}")
        print(f"Total Iterations: {self.iteration}")
        
        if self.best_config:
            print(f"\n🏆 BEST CONFIGURATION:")
            print(f"   Profit Weight: {self.best_config.profit:.3f}")
            print(f"   Loss Penalty: {self.best_config.loss_penalty:.3f}")
            print(f"   Bust Penalty: {self.best_config.bust_penalty:.3f}")
            
            if self.best_config.close_call_bonus > 0:
                print(f"   Close Call Bonus: {self.best_config.close_call_bonus:.3f}")
            if self.best_config.dealer_weak_bonus > 0:
                print(f"   Dealer Weak Bonus: {self.best_config.dealer_weak_bonus:.3f}")
            if self.best_config.aggressive_hit_bonus > 0:
                print(f"   Aggressive Hit Bonus: {self.best_config.aggressive_hit_bonus:.3f}")
            
            # Save configuration
            Path("outputs/auto_tune").mkdir(parents=True, exist_ok=True)
            
            with open("outputs/auto_tune/best_config.json", "w") as f:
                json.dump({
                    "win_rate": self.best_win_rate,
                    "config": self.best_config.to_dict(),
                    "iterations": self.iteration
                }, f, indent=2)
            
            with open("outputs/auto_tune/history.pkl", "wb") as f:
                pickle.dump(self.history, f)
            
            print(f"\n💾 Results saved to outputs/auto_tune/")
            
            print(f"\n🎮 To use this configuration:")
            print(f"python train.py --mode custom \\")
            for i, arg in enumerate(self.best_config.to_cmd_args()):
                if i % 2 == 0:
                    print(f"  {arg}", end=" ")
                else:
                    print(arg, end=" \\\n" if i < len(self.best_config.to_cmd_args())-1 else "\n")

if __name__ == "__main__":
    tuner = HyperparameterTuner(target_win_rate=0.45, max_iterations=50)
    tuner.run()
