# Separate Critics Per Reward Objective

## Overview

This implementation investigates the use of **separate critics per reward term** in multi-objective reinforcement learning. This approach helps prevent the policy from exploiting easy objectives where it performs well while ignoring other important signals.

## Motivation

In traditional multi-objective RL with a single combined reward, the policy may:
- Focus heavily on objectives that are easier to optimize
- Ignore objectives that are harder to learn but equally important
- Create an imbalance in how different objectives are represented in the value function

By using **separate critics** for each objective, we ensure that:
1. Each objective gets its own dedicated value function estimate
2. The policy is explicitly evaluated on each objective separately
3. Advantages are computed per-objective, preventing one objective from dominating
4. The policy must perform well across all objectives, not just the easiest ones

## Architecture

### MultiObjectiveActorCritic Network

The `MultiObjectiveActorCritic` network extends the standard Actor-Critic architecture:

- **Shared Feature Extractor**: Common layers that process the state
- **Single Actor Head**: One policy network that outputs action probabilities
- **Multiple Critic Heads**: One value function per objective
  - `critic_expected_return`: Estimates value for expected return objective
  - `critic_sharpe_ratio`: Estimates value for Sharpe ratio objective
  - `critic_win_rate`: Estimates value for win rate objective

### MultiObjectivePPO Algorithm

The `MultiObjectivePPO` class implements PPO with separate critics:

1. **Separate Reward Storage**: Stores rewards for each objective separately
2. **Separate Value Storage**: Stores value estimates for each objective separately
3. **Per-Objective GAE**: Computes Generalized Advantage Estimation (GAE) for each objective independently
4. **Weighted Advantage Combination**: Combines advantages using objective weights
5. **Per-Objective Value Loss**: Computes value loss for each critic separately, then combines with weights

## Key Differences from Standard PPO

| Aspect | Standard PPO | MultiObjectivePPO |
|--------|-------------|-------------------|
| Value Function | Single critic for combined reward | Separate critic per objective |
| Reward Storage | Single combined reward | Separate rewards per objective |
| Advantage Computation | Single GAE for combined reward | Separate GAE per objective, then weighted combination |
| Value Loss | Single MSE loss | Weighted sum of per-objective MSE losses |
| Policy Update | Uses combined advantage | Uses weighted combination of per-objective advantages |

## Usage

### Configuration

Enable separate critics in `configs/config.yaml`:

```yaml
objectives:
  weights:
    expected_return: 0.5
    sharpe_ratio: 0.3
    win_rate: 0.2
  use_separate_critics: true  # Enable separate critics for PPO
```

### Training

When training with PPO, the system will automatically use `MultiObjectivePPO` if `use_separate_critics: true`:

```bash
python src/training/train.py --algorithm ppo --config configs/config.yaml
```

### How It Works

1. **During Rollout**:
   - For each step, the agent receives separate rewards for each objective
   - Each critic estimates the value for its specific objective
   - All values and rewards are stored separately

2. **During Training**:
   - GAE is computed separately for each objective using its own rewards and values
   - Advantages are normalized per-objective
   - Advantages are combined using objective weights: `A_combined = Σ w_i * A_i`
   - Policy loss uses the combined advantage
   - Value loss is computed per-objective and combined: `L_value = Σ w_i * L_i`

## Benefits

1. **Prevents Objective Exploitation**: The policy cannot ignore difficult objectives since each has its own value function
2. **Better Balance**: Forces the policy to consider all objectives during learning
3. **Interpretability**: Can inspect value estimates for each objective separately
4. **Flexible Weighting**: Objective weights can be adjusted without retraining value functions

## Limitations

1. **Increased Model Size**: More parameters due to multiple critic heads
2. **More Complex Training**: Requires storing and managing separate rewards/values
3. **Computational Overhead**: Slightly more computation during forward/backward passes

## Implementation Details

### Network Architecture

```python
MultiObjectiveActorCritic(
    state_dim=state_dim,
    action_dim=action_dim,
    objective_names=['expected_return', 'sharpe_ratio', 'win_rate'],
    hidden_dims=[128, 128]
)
```

### Training Flow

```python
# 1. Store transition with separate rewards/values
agent.store_transition(
    state, action,
    objective_rewards={'expected_return': r1, 'sharpe_ratio': r2, 'win_rate': r3},
    log_prob, 
    objective_values={'expected_return': v1, 'sharpe_ratio': v2, 'win_rate': v3},
    done
)

# 2. Compute per-objective advantages
for obj_name in objective_names:
    advantages[obj_name], returns[obj_name] = compute_gae_per_objective(
        rewards[obj_name], values[obj_name], dones
    )

# 3. Combine advantages
combined_advantages = sum(weights[obj] * advantages[obj] for obj in objective_names)

# 4. Update policy and critics
policy_loss = compute_policy_loss(combined_advantages)
value_loss = sum(weights[obj] * value_losses[obj] for obj in objective_names)
```

## Experimental Comparison

To compare standard PPO vs. MultiObjectivePPO:

1. Train with `use_separate_critics: false` (baseline)
2. Train with `use_separate_critics: true` (separate critics)
3. Compare:
   - Final performance on each objective
   - Balance across objectives
   - Training stability
   - Convergence speed

## References

This approach is inspired by research on multi-objective reinforcement learning and multi-task learning, where separate value functions help prevent task interference and ensure balanced learning across objectives.

