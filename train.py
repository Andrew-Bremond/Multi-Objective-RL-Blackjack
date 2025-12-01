import argparse
import json
from pathlib import Path

import torch

from src.config import PPOHyperParams, RewardWeights, TrainSettings, preset_weights
from src.envs import make_env
from src.training import evaluate_agent, train_loop


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-objective PPO for Blackjack")
    parser.add_argument("--mode", type=str, default="baseline", 
                       help="baseline | risk_averse | strategic | optimal | custom")
    parser.add_argument("--profit-weight", type=float, default=1.0)
    parser.add_argument("--loss-penalty", type=float, default=0.0)
    parser.add_argument("--bust-penalty", type=float, default=0.0)
    # New objective weights
    parser.add_argument("--close-call-bonus", type=float, default=0.0)
    parser.add_argument("--dealer-weak-bonus", type=float, default=0.0)
    parser.add_argument("--conservative-stand-bonus", type=float, default=0.0)
    parser.add_argument("--aggressive-hit-bonus", type=float, default=0.0)
    parser.add_argument("--perfect-play-bonus", type=float, default=0.0)
    # Additional objectives
    parser.add_argument("--blackjack-bonus", type=float, default=0.0)
    parser.add_argument("--push-penalty", type=float, default=0.0)
    parser.add_argument("--early-stand-penalty", type=float, default=0.0)
    parser.add_argument("--dealer-bust-bonus", type=float, default=0.0)

    parser.add_argument("--total-episodes", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rollout-length", type=int, default=2048)
    parser.add_argument("--update-epochs", type=int, default=10)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--entropy-coef", type=float, default=0.05)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)

    parser.add_argument("--eval-episodes", type=int, default=1000)
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to load checkpoint for resume/eval")
    parser.add_argument("--eval-only", action="store_true", help="Skip training and only evaluate checkpoint")
    parser.add_argument("--deterministic-eval", action="store_true", help="Use argmax actions during evaluation")
    return parser.parse_args()


def build_weights(args) -> RewardWeights:
    if args.mode.lower() == "custom":
        return RewardWeights(
            profit=args.profit_weight,
            loss_penalty=args.loss_penalty,
            bust_penalty=args.bust_penalty,
            close_call_bonus=args.close_call_bonus,
            dealer_weak_bonus=args.dealer_weak_bonus,
            conservative_stand_bonus=args.conservative_stand_bonus,
            aggressive_hit_bonus=args.aggressive_hit_bonus,
            perfect_play_bonus=args.perfect_play_bonus,
            blackjack_bonus=args.blackjack_bonus,
            push_penalty=args.push_penalty,
            early_stand_penalty=args.early_stand_penalty,
            dealer_bust_bonus=args.dealer_bust_bonus
        )
    return preset_weights(args.mode)


def main():
    args = parse_args()
    weights = build_weights(args)

    hyperparams = PPOHyperParams(
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_coef=args.clip_coef,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        learning_rate=args.learning_rate,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        rollout_length=args.rollout_length,
        max_grad_norm=args.max_grad_norm,
    )

    settings = TrainSettings(
        total_episodes=args.total_episodes,
        seed=args.seed,
        eval_episodes=args.eval_episodes,
        deterministic_eval=args.deterministic_eval,
    )

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else None
    save_dir = Path(args.save_dir) if args.save_dir else Path("outputs") / args.mode

    if args.eval_only:
        if not checkpoint_path or not checkpoint_path.exists():
            raise FileNotFoundError("Provide --checkpoint pointing to a trained model when using --eval-only.")
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        weights = RewardWeights(**ckpt.get("weights", weights.__dict__))
        hyper_data = ckpt.get("hyperparams")
        if hyper_data:
            hyperparams = PPOHyperParams(**hyper_data)

        env = make_env(weights, seed=settings.seed)
        from src.ppo_agent import PPOAgent  # Local import to avoid cycle on eval-only

        agent = PPOAgent(
            obs_dim=3,
            action_dim=2,
            lr=hyperparams.learning_rate,
            gamma=hyperparams.gamma,
            gae_lambda=hyperparams.gae_lambda,
            clip_coef=hyperparams.clip_coef,
            entropy_coef=hyperparams.entropy_coef,
            value_coef=hyperparams.value_coef,
            max_grad_norm=hyperparams.max_grad_norm,
        )
        agent.model.load_state_dict(ckpt["model_state"])
        results = evaluate_agent(env, agent, episodes=args.eval_episodes, deterministic=args.deterministic_eval)
        print(json.dumps(results, indent=2))
        return

    results = train_loop(
        weights=weights,
        hyperparams=hyperparams,
        settings=settings,
        save_dir=save_dir,
        checkpoint_path=checkpoint_path,
    )
    print("Final evaluation:", json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
