from __future__ import annotations

from pathlib import Path
from typing import List

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np

from .envs import make_env, normalize_obs
from .ppo_agent import PPOAgent


def _frame_as_array(text_lines: List[str]) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.axis("off")
    ax.text(0.02, 0.95, "\n".join(text_lines), va="top", ha="left", fontsize=11, family="monospace")
    fig.tight_layout()
    fig.canvas.draw()
    frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return frame


def generate_policy_gif(agent: PPOAgent, weights, episodes: int, out_path: Path, seed: int = 123):
    env = make_env(weights, seed=seed)
    frames: List[np.ndarray] = []

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        step_idx = 0
        while not done:
            obs_norm = normalize_obs(obs)
            action = agent.select_action(obs_norm, deterministic=True)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step_idx += 1

            player_sum, dealer_card, usable_ace = obs
            text_lines = [
                f"Episode {ep + 1}  Step {step_idx}",
                f"Player sum: {player_sum}",
                f"Dealer shows: {dealer_card}",
                f"Usable ace: {usable_ace}",
                f"Action: {'Hit' if action == 1 else 'Stay'}",
            ]
            if done:
                if info.get("is_win", False):
                    outcome = "WIN"
                elif info.get("is_loss", False):
                    outcome = "LOSS"
                else:
                    outcome = "DRAW"
                text_lines.append(f"Outcome: {outcome}")
                text_lines.append(f"Bust: {info.get('is_bust', False)}")
            frames.append(_frame_as_array(text_lines))
            obs = next_obs

    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_path, frames, duration=0.5)
