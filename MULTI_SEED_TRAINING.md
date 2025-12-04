# Multi-Seed Training Guide

## Overview

The training system now supports running multiple seeds automatically and organizing results in a structured directory format. Results are aggregated across seeds with averaged plots and best model identification.

## Directory Structure

Results are organized as follows:

```
results/
├── dqn/
│   ├── dqn_5_20251204_130710/
│   │   ├── best_model.pth
│   │   ├── training_metrics.csv
│   │   ├── eval_metrics.csv
│   │   ├── summary.json
│   │   └── ...
│   ├── dqn_10_20251204_130715/
│   │   └── ...
│   ├── dqn_15_20251204_130720/
│   │   └── ...
│   ├── dqn_aggregated_results.png
│   ├── dqn_multi_objective_aggregated.png
│   └── dqn_best_model_summary.txt
├── ddqn/
│   └── ...
├── dueling_ddqn/
│   └── ...
└── ppo/
    └── ...
```

## Configuration

In `configs/config.yaml`, set:

```yaml
training:
  num_episodes: 100000
  seed: [5, 10, 15]  # List of seeds to run
```

Or use a single seed:
```yaml
training:
  seed: 42  # Single seed (will be converted to list)
```

## Usage

### Training with Multiple Seeds

Simply run the training script as usual:

```bash
python src/training/train.py --algorithm ppo --config configs/config.yaml
```

The script will:
1. Read seeds from config (list or single value)
2. Train each seed sequentially
3. Save results in `results/{algorithm}/{algorithm}_{seed}_{timestamp}/`
4. Automatically aggregate results after all seeds complete

### Manual Aggregation

If you need to aggregate results separately (e.g., after training):

```bash
python src/training/aggregate_results.py --algorithm ppo --results_dir results/ppo
```

## Output Files

### Per-Seed Directories

Each seed gets its own directory with:
- `best_model.pth` - Best model checkpoint
- `training_metrics.csv` - Training metrics over time
- `eval_metrics.csv` - Evaluation metrics
- `summary.json` - Summary statistics
- Individual plots (training, eval, multi-objective)

### Aggregated Files (in `results/{algorithm}/`)

1. **`{algorithm}_aggregated_results.png`**
   - 2x2 plot showing:
     - Training reward (averaged with std dev bands)
     - Evaluation reward (averaged)
     - Expected return (averaged)
     - Sharpe ratio & Win rate (averaged)

2. **`{algorithm}_multi_objective_aggregated.png`**
   - 1x3 plot showing averaged multi-objective metrics:
     - Expected Return
     - Sharpe Ratio
     - Win Rate

3. **`{algorithm}_best_model_summary.txt`**
   - Text file identifying the best model across all seeds
   - Includes metrics for all seeds
   - Best model selected by highest evaluation reward

## Example

```bash
# Train PPO with 3 seeds (5, 10, 15) for 100,000 episodes
python src/training/train.py --algorithm ppo --config configs/config.yaml

# This will create:
# - results/ppo/ppo_5_.../
# - results/ppo/ppo_10_.../
# - results/ppo/ppo_15_.../
# - results/ppo/ppo_aggregated_results.png
# - results/ppo/ppo_multi_objective_aggregated.png
# - results/ppo/ppo_best_model_summary.txt
```

## Notes

- Seeds are trained sequentially (one after another)
- Each seed maintains its own random state
- Aggregation happens automatically after all seeds complete
- If aggregation fails, you can run it manually using the command above
- The best model is selected based on highest evaluation reward across all seeds

