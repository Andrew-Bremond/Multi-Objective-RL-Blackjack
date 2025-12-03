"""Training script for DQN and Dueling DQN on Blackjack."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.config import RewardWeights, preset_weights
from src.envs import make_env, normalize_obs
from src.dqn_agent import DDQNAgent


def parse_args():
    parser = argparse.ArgumentParser(description="DQN/Dueling DQN for Blackjack")
    
    # Algorithm selection
    parser.add_argument("--algorithm", type=str, default="dueling", choices=["dqn", "dueling"],
                       help="DQN algorithm: 'dqn' for standard DDQN, 'dueling' for Dueling DDQN")
    
    # Reward mode
    parser.add_argument("--mode", type=str, default="optimal", 
                       help="baseline | risk_averse | strategic | optimal | custom")
    parser.add_argument("--profit-weight", type=float, default=1.0)
    parser.add_argument("--loss-penalty", type=float, default=0.0)
    parser.add_argument("--bust-penalty", type=float, default=0.0)
    parser.add_argument("--close-call-bonus", type=float, default=0.0)
    
    # Training parameters
    parser.add_argument("--total-episodes", type=int, default=50000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.01)
    parser.add_argument("--epsilon-decay", type=float, default=0.9995)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--buffer-capacity", type=int, default=100000)
    parser.add_argument("--target-update-freq", type=int, default=500)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    
    # Evaluation
    parser.add_argument("--eval-episodes", type=int, default=1000)
    parser.add_argument("--eval-interval", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-dir", type=str, default=None)
    
    return parser.parse_args()


def build_weights(args) -> RewardWeights:
    if args.mode.lower() == "custom":
        return RewardWeights(
            profit=args.profit_weight,
            loss_penalty=args.loss_penalty,
            bust_penalty=args.bust_penalty,
            close_call_bonus=args.close_call_bonus
        )
    return preset_weights(args.mode)


def evaluate_agent(env, agent: DDQNAgent, episodes: int) -> dict:
    """Evaluate agent performance."""
    wins = losses = busts = 0
    total_reward = 0.0
    
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        
        while not done:
            obs_norm = normalize_obs(obs)
            action = agent.select_action(obs_norm, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        
        wins += int(info.get("is_win", False))
        losses += int(info.get("is_loss", False))
        busts += int(info.get("is_bust", False))
        total_reward += float(info.get("base_reward", reward))
    
    total = max(1, episodes)
    return {
        "win_rate": wins / total,
        "loss_rate": losses / total,
        "bust_rate": busts / total,
        "avg_reward": total_reward / total,
    }


def train():
    args = parse_args()
    
    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create environments
    weights = build_weights(args)
    train_env = make_env(weights, seed=args.seed)
    eval_env = make_env(weights, seed=args.seed + 1000)
    
    # Create agent
    use_dueling = (args.algorithm == "dueling")
    agent = DDQNAgent(
        obs_dim=3,
        action_dim=2,
        use_dueling=use_dueling,
        lr=args.learning_rate,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        buffer_capacity=args.buffer_capacity,
        batch_size=args.batch_size,
        target_update_freq=args.target_update_freq,
    )
    
    # Setup save directory
    if args.save_dir:
        save_dir = Path(args.save_dir)
    else:
        save_dir = Path("outputs") / f"{args.algorithm}_{args.mode}"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    print(f"\n{'='*60}")
    print(f"Training {args.algorithm.upper()} on Blackjack")
    print(f"Target: 45% win rate")
    print(f"{'='*60}\n")
    
    episode = 0
    step = 0
    history = []
    best_win_rate = 0.0
    
    pbar = tqdm(total=args.total_episodes, desc="Training", unit="ep")
    
    obs, _ = train_env.reset()
    
    while episode < args.total_episodes:
        obs_norm = normalize_obs(obs)
        
        # Select action
        action = agent.select_action(obs_norm, deterministic=False)
        
        # Step environment
        next_obs, reward, terminated, truncated, info = train_env.step(action)
        done = terminated or truncated
        next_obs_norm = normalize_obs(next_obs)
        
        # Store experience
        agent.store_experience(obs_norm, action, reward, next_obs_norm, done)
        
        # Update agent (after warmup)
        if step >= args.warmup_steps:
            metrics = agent.update()
        
        obs = next_obs
        step += 1
        
        if done:
            episode += 1
            pbar.update(1)
            obs, _ = train_env.reset()
            
            # Periodic evaluation
            if episode % args.eval_interval == 0:
                eval_results = evaluate_agent(eval_env, agent, args.eval_episodes)
                
                pbar.write(f"\n{'='*60}")
                pbar.write(f"Episode {episode}/{args.total_episodes}")
                pbar.write(f"  Win Rate: {eval_results['win_rate']:.1%}")
                pbar.write(f"  Bust Rate: {eval_results['bust_rate']:.1%}")
                pbar.write(f"  Avg Reward: {eval_results['avg_reward']:.3f}")
                pbar.write(f"  Epsilon: {agent.epsilon:.3f}")
                pbar.write(f"{'='*60}\n")
                
                history.append({
                    "episode": episode,
                    **eval_results,
                    "epsilon": agent.epsilon
                })
                
                # Save best model
                if eval_results['win_rate'] > best_win_rate:
                    best_win_rate = eval_results['win_rate']
                    torch.save({
                        'q_network': agent.q_network.state_dict(),
                        'target_network': agent.target_network.state_dict(),
                        'optimizer': agent.optimizer.state_dict(),
                        'episode': episode,
                        'win_rate': best_win_rate,
                        'config': vars(args),
                        'weights': weights.__dict__
                    }, save_dir / "best_model.pt")
                    
                    if best_win_rate >= 0.45:
                        pbar.write(f"\n🎉 TARGET ACHIEVED! {best_win_rate:.1%} win rate!")
    
    pbar.close()
    
    # Final evaluation
    print(f"\n{'='*60}")
    print("FINAL EVALUATION")
    print(f"{'='*60}")
    
    final_results = evaluate_agent(eval_env, agent, args.eval_episodes * 2)
    
    print(f"\nAlgorithm: {args.algorithm.upper()}")
    print(f"Win Rate: {final_results['win_rate']:.1%}")
    print(f"Loss Rate: {final_results['loss_rate']:.1%}")
    print(f"Bust Rate: {final_results['bust_rate']:.1%}")
    print(f"Avg Reward: {final_results['avg_reward']:.3f}")
    print(f"\nBest Win Rate During Training: {best_win_rate:.1%}")
    
    # Save results
    with open(save_dir / "final_results.json", "w") as f:
        json.dump({
            "final": final_results,
            "best_win_rate": best_win_rate,
            "algorithm": args.algorithm,
            "config": vars(args),
            "history": history
        }, f, indent=2)
    
    print(f"\n✅ Results saved to: {save_dir}")
    
    if final_results['win_rate'] >= 0.45:
        print(f"\n🎉🎉🎉 SUCCESS! Achieved {final_results['win_rate']:.1%} win rate!")
    elif final_results['win_rate'] >= 0.44:
        print(f"\n🔥 Very close! {final_results['win_rate']:.1%} win rate (only {0.45 - final_results['win_rate']:.1%} from target)")
    
    return final_results


if __name__ == "__main__":
    train()

