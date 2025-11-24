from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .config import PPOHyperParams, RewardWeights, TrainSettings
from .envs import make_env, normalize_obs
from .ppo_agent import PPOAgent, Rollout


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def collect_rollout(env, agent: PPOAgent, rollout_length: int) -> Tuple[Rollout, Dict[str, float]]:
    observations: List[np.ndarray] = []
    actions: List[int] = []
    log_probs: List[float] = []
    rewards: List[float] = []
    dones: List[float] = []
    values: List[float] = []

    episode_counter = 0
    wins = 0
    losses = 0
    busts = 0
    base_reward_sum = 0.0

    obs, _ = env.reset()
    obs_norm = normalize_obs(obs)

    for _ in range(rollout_length):
        action, log_prob, value = agent.act(obs_norm)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        observations.append(obs_norm)
        actions.append(action)
        log_probs.append(log_prob)
        rewards.append(reward)
        dones.append(float(done))
        values.append(value)

        if done:
            episode_counter += 1
            wins += int(info.get("is_win", False))
            losses += int(info.get("is_loss", False))
            busts += int(info.get("is_bust", False))
            base_reward_sum += float(info.get("base_reward", reward))
            next_obs, _ = env.reset()

        obs_norm = normalize_obs(next_obs)

    with torch.no_grad():
        obs_tensor = torch.from_numpy(obs_norm).float().to(agent.device).unsqueeze(0)
        _, last_value_tensor = agent.model(obs_tensor)
        last_value = float(last_value_tensor.squeeze().item())

    rollout = Rollout(
        observations=torch.tensor(np.array(observations), dtype=torch.float32),
        actions=torch.tensor(actions, dtype=torch.long),
        log_probs=torch.tensor(log_probs, dtype=torch.float32),
        rewards=torch.tensor(rewards, dtype=torch.float32),
        dones=torch.tensor(dones, dtype=torch.float32),
        values=torch.tensor(values, dtype=torch.float32),
        last_value=last_value,
    )

    stats = {
        "episodes": episode_counter,
        "wins": wins,
        "losses": losses,
        "busts": busts,
        "base_reward_sum": base_reward_sum,
    }

    return rollout, stats


def evaluate_agent(env, agent: PPOAgent, episodes: int, deterministic: bool = True) -> Dict[str, float]:
    wins = losses = busts = 0
    total_reward = 0.0
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            obs_norm = normalize_obs(obs)
            action = agent.select_action(obs_norm, deterministic=deterministic)
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


def plot_metrics(df: pd.DataFrame, save_path: Path):
    plt.figure(figsize=(8, 5))
    plt.plot(df["episodes"], df["win_rate"], label="Win rate")
    plt.plot(df["episodes"], df["loss_rate"], label="Loss rate")
    plt.plot(df["episodes"], df["bust_rate"], label="Bust rate")
    plt.xlabel("Episodes seen")
    plt.ylabel("Rate")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def train_loop(
    weights: RewardWeights,
    hyperparams: PPOHyperParams,
    settings: TrainSettings,
    save_dir: Path,
    checkpoint_path: Path | None = None,
) -> Dict[str, float]:
    set_seed(settings.seed)
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "checkpoints").mkdir(exist_ok=True)

    env = make_env(weights, seed=settings.seed)
    eval_env = make_env(weights, seed=settings.seed + 1000)

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

    if checkpoint_path and checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location=agent.device)
        agent.model.load_state_dict(state["model_state"])
        agent.optimizer.load_state_dict(state["optimizer_state"])

    history: List[Dict[str, float]] = []
    episodes_seen = 0
    pbar = tqdm(total=settings.total_episodes, desc="Training episodes", unit="ep")

    while episodes_seen < settings.total_episodes:
        rollout, stats = collect_rollout(env, agent, hyperparams.rollout_length)
        losses = agent.update(
            rollout=rollout,
            update_epochs=hyperparams.update_epochs,
            minibatch_size=hyperparams.minibatch_size,
        )

        episodes_seen += stats["episodes"]
        pbar.update(stats["episodes"])

        ep_count = max(1, stats["episodes"])
        win_rate = stats["wins"] / ep_count
        loss_rate = stats["losses"] / ep_count
        bust_rate = stats["busts"] / ep_count
        avg_reward = stats["base_reward_sum"] / ep_count

        history.append(
            {
                "episodes": episodes_seen,
                "win_rate": win_rate,
                "loss_rate": loss_rate,
                "bust_rate": bust_rate,
                "avg_reward": avg_reward,
                "policy_loss": losses["policy_loss"],
                "value_loss": losses["value_loss"],
                "entropy": losses["entropy"],
            }
        )

    pbar.close()

    metrics_df = pd.DataFrame(history)
    metrics_path = save_dir / "metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    plot_metrics(metrics_df, save_path=save_dir / "learning_curve.png")

    # Final evaluation
    eval_results = evaluate_agent(eval_env, agent, episodes=settings.eval_episodes, deterministic=settings.deterministic_eval)
    with open(save_dir / "eval_summary.json", "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)

    checkpoint_out = save_dir / "checkpoints" / "ppo_blackjack.pt"
    torch.save(
        {
            "model_state": agent.model.state_dict(),
            "optimizer_state": agent.optimizer.state_dict(),
            "weights": asdict(weights),
            "hyperparams": asdict(hyperparams),
            "train_settings": asdict(settings),
        },
        checkpoint_out,
    )

    return eval_results
