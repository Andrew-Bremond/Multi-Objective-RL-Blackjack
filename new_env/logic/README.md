# Multi-Objective RL for Blackjack

This project implements four reinforcement learning algorithms (DQN, Dueling DDQN, DDQN, and PPO) for a custom Blackjack environment with doubling and bet sizing capabilities, using multi-objective optimization.

## Features

- **Custom Blackjack Environment**: Extended gymnasium Blackjack-v1 with:

  - Two-phase game structure (betting phase + playing phase)
  - Doubling down action
  - Discrete bet sizing (1x, 2x, 3x base bet)

- **Multi-Objective Optimization**: Three objectives:

  1. Expected Return - Maximize average profit per game
  2. Risk-Adjusted Return (Sharpe Ratio) - Balance return with volatility
  3. Win Rate - Maximize percentage of winning games

- **Four RL Algorithms**:
  - DQN (Deep Q-Network)
  - DDQN (Double DQN)
  - Dueling DDQN
  - PPO (Proximal Policy Optimization)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training

```bash
python src/training/train.py --algorithm dqn --config configs/config.yaml
```

Available algorithms: `dqn`, `ddqn`, `dueling_ddqn`, `ppo`

### Evaluation

```bash
python src/training/evaluate.py --model_path results/models/best_model.pth
```

## Project Structure

```
logic/
├── requirements.txt
├── README.md
├── src/
│   ├── environment/          # Custom Blackjack environment
│   ├── algorithms/           # RL algorithm implementations
│   ├── objectives/           # Multi-objective framework
│   ├── utils/                # Utilities (networks, replay buffer)
│   └── training/             # Training and evaluation scripts
├── configs/                  # Configuration files
└── results/                  # Model outputs, logs, plots
```

## Configuration

Edit `configs/config.yaml` to adjust hyperparameters, environment settings, and multi-objective weights.

## Results

After training, results are saved in `results/`:

- Best model checkpoint
- Training/evaluation metrics (CSV)
- Visualization plots (PNG)
- Training summary report
