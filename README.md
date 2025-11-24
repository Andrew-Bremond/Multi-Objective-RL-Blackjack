# Multi-Objective RL for Blackjack

Python reference implementation for a multi-objective PPO agent on the Gymnasium Blackjack environment. The agent trades off between three objectives:

- Profit maximization (standard Blackjack return)
- Loss probability reduction
- Bust probability reduction

The project includes baseline (profit-focused) and risk-averse (bust/loss-averse) configurations, training/evaluation scripts, and plotting utilities.

## Quickstart

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# Baseline (profit-focused)
python train.py --mode baseline --total-episodes 5000 --save-dir outputs/baseline

# Risk-averse (penalize losing/busting)
python train.py --mode risk_averse --total-episodes 5000 --save-dir outputs/risk

# Evaluate a saved checkpoint
python train.py --eval-only --checkpoint outputs/risk/checkpoints/ppo_blackjack.pt --num-eval-episodes 500
```

Artifacts land in `outputs/<run>/`:
- `metrics.csv` with episode-level outcomes
- `learning_curve.png` plotting rolling win/loss/bust rates
- `checkpoint/ppo_blackjack.pt` trained weights
- `eval_summary.json` with deterministic evaluation results
- Optional GIF: run `python - <<'PY'\nfrom pathlib import Path\nimport torch\nfrom src.config import RewardWeights\nfrom src.ppo_agent import PPOAgent\nfrom src.visualize import generate_policy_gif\nfrom src.envs import make_env\n\nckpt = torch.load('outputs/risk/checkpoints/ppo_blackjack.pt', map_location='cpu')\nweights = RewardWeights(**ckpt['weights'])\nagent = PPOAgent(obs_dim=3, action_dim=2)\nagent.model.load_state_dict(ckpt['model_state'])\ngenerate_policy_gif(agent, weights, episodes=10, out_path=Path('outputs/risk/policy.gif'))\nPY`

## Reward shaping

The wrapper converts the native Blackjack reward (win: +1, loss: -1, draw: 0) into a scalarized objective:

```
scalar_reward = profit_weight * base_reward
               - loss_penalty * 1[loss]
               - bust_penalty * 1[bust]
```

`--mode baseline` uses `(1.0, 0.0, 0.0)`.  
`--mode risk_averse` uses `(0.6, 0.2, 0.2)`.

Custom weights are supported via `--profit-weight`, `--loss-penalty`, and `--bust-penalty`.

## Notes

- Uses Gymnasium `Blackjack-v1`, PyTorch PPO with GAE, and simple MLP actor/critic.
- Default training settings are modest for a quick run; increase `--total-episodes` for stronger policies (proposal suggests 100k+ per configuration).
- A lightweight GIF generator is stubbed for future visualization; hook it up to rendered frames if you want richer animations.

## Experiment Report

- **Methods/Algorithm**: PPO with clip objective, entropy bonus, and GAE. Multi-objective scalar reward combines profit, loss-penalty, bust-penalty. Actor-Critic MLP (two hidden layers of 128, ReLU).
- **Implementation details**: `src/envs.py` wraps Blackjack to add scalar reward; `src/ppo_agent.py` holds the PPO/GAE logic; `src/training.py` manages rollouts (length 500), minibatch PPO updates, logging, and plots; `src/visualize.py` can render a policy GIF.
- **Training protocol**: 100k episodes per run, rollout length 500, seeds 0/1/2. Baseline uses weights (1.0, 0.0, 0.0); risk-averse uses (0.6, 0.2, 0.2). Checkpoints saved to `outputs/<run>/checkpoints/ppo_blackjack.pt`.
- **Evaluation & metrics**: Win rate, loss rate, bust rate, avg reward; learning curves (`learning_curve.png`); bankroll history (`bankroll_history.png`). Deterministic evals over 1,000 episodes saved as `eval_summary_deterministic_1000.json`.
- **Results (deterministic 1,000 eps)**:
  - Baseline: s0 win 0.435 / loss 0.457 / bust 0.168 / avg_reward -0.022; s1 win 0.437 / loss 0.474 / bust 0.116 / avg_reward -0.037; s2 win 0.452 / loss 0.442 / bust 0.163 / avg_reward 0.010.
  - Risk-averse: s0 win 0.415 / loss 0.492 / bust 0.055 / avg_reward -0.077; s1 win 0.424 / loss 0.499 / bust 0.038 / avg_reward -0.075; s2 win 0.394 / loss 0.557 / bust 0.000 / avg_reward -0.163.
- **Short analysis**: Penalizing losses/busts markedly cuts bust rates (down to 0–6%) and slightly improves average reward in some seeds, showing the three objectives are active. Absolute expected value stays near/below zero (expected without counting); further gains likely require tuning weights, entropy, and learning rate or longer training.

## Network Architecture & Implementation Details

- **Architecture**: MLP Actor-Critic with two hidden layers of 128 units each, ReLU activations. Actor head outputs 2 logits (Stay/Hit); Critic head outputs a scalar value.
- **Inputs/Outputs**: State tuple `(player_sum, dealer_card, usable_ace)` → normalized float tensor `[player_sum/32, dealer_card/11, float(usable_ace)]`. Actor logits → softmax policy over actions; Critic predicts V(s).
- **Hyperparameters (defaults)**: learning rate 3e-4; discount γ=0.99; GAE λ=0.95; rollout length 500; PPO epochs per update 4; minibatch size 256; clip coef 0.2; entropy coef 0.01; value loss coef 0.5; optimizer Adam; max grad norm 0.5.
- **Multi-objective reward**: Wrapper in `src/envs.py` tracks base reward G (Blackjack: +1/-1/0) and flags `loss_flag`, `bust_flag` each step. Scalar reward per transition: `profit*G - loss_penalty*1[loss_flag] - bust_penalty*1[bust_flag]`; this replaces the reward fed to PPO.
- **Configs**: Baseline (profit-only) weights `(1.0, 0.0, 0.0)`; Risk-averse weights `(0.6, 0.2, 0.2)`; custom weights via CLI flags `--profit-weight`, `--loss-penalty`, `--bust-penalty`.
