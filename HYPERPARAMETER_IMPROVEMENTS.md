# Hyperparameter Improvements for Multi-Objective RL Blackjack

## Why Training Was So Fast

Blackjack episodes are **extremely short** (3-5 steps per game on average):
- Initial cards dealt
- Player decides: hit or stand (1-3 times typically)
- Game ends

With a rollout length of 2048 steps, you're collecting **~400-600 episodes per rollout**. 
So 5,000 episodes only requires ~10-15 rollout collections, completing in 1-2 seconds!

## Previous Performance Issues

The agent was **losing money consistently**:
- ❌ Win rate: ~34-35% (should be 42-49% with optimal strategy)
- ❌ Loss rate: ~58-59%
- ✅ Bust rate: 6-7% (acceptable)
- ❌ Average reward: **-0.24** (losing 24 cents per game)

## Key Improvements Made

### 1. **Increased Entropy Coefficient: 0.01 → 0.05**
- **Why**: The agent needs more exploration to discover optimal hit/stand decisions
- **Impact**: Better exploration of the action space, especially early in training

### 2. **Increased Learning Rate: 3e-4 → 5e-4**
- **Why**: Faster convergence without instability (Blackjack is a simple environment)
- **Impact**: Agent learns patterns more quickly

### 3. **More Update Epochs: 4 → 10**
- **Why**: Better utilization of collected data, more thorough policy updates
- **Impact**: More stable learning from each rollout

### 4. **Deeper Network: [128, 128] → [256, 256, 128]**
- **Why**: More capacity to learn complex decision boundaries
- **Impact**: Better representation of state-action value functions
- **Note**: Added 3rd hidden layer for more non-linear transformations

### 5. **Increased Training Episodes: 5,000 → 50,000**
- **Why**: Previous training was too short for convergence
- **Impact**: ~10x more training data (still only takes ~10-20 seconds!)
- **Note**: This is now the default, but can be overridden with `--total-episodes`

### 6. **More Evaluation Episodes: 500 → 1,000**
- **Why**: More accurate performance metrics with lower variance
- **Impact**: Better assessment of true agent performance

## Expected Results

With these improvements, you should see:
- **Win rate**: 40-48%
- **Loss rate**: 45-52%
- **Bust rate**: 5-10%
- **Average reward**: -0.05 to +0.05 (near break-even or slight profit)

Note: Perfect play in standard Blackjack has ~49% win rate due to house edge.

## Usage

### Quick Test with Improved Defaults
```bash
python3 train.py --mode baseline
```

### Custom Training
```bash
# Baseline with explicit settings
python3 train.py \
  --mode baseline \
  --total-episodes 50000 \
  --learning-rate 5e-4 \
  --entropy-coef 0.05 \
  --update-epochs 10

# Risk-averse strategy
python3 train.py \
  --mode risk_averse \
  --total-episodes 100000 \
  --save-dir outputs/risk_averse_v2
```

### For Even Better Results (Longer Training)
```bash
python3 train.py \
  --mode baseline \
  --total-episodes 100000 \
  --learning-rate 3e-4 \
  --entropy-coef 0.03 \
  --update-epochs 15 \
  --save-dir outputs/baseline_extended
```

## Training Time Estimates

- 50,000 episodes: ~10-20 seconds
- 100,000 episodes: ~20-40 seconds
- 200,000 episodes: ~40-80 seconds

Still very fast! 🚀

## Monitoring Training

Watch for these positive signs:
1. **Win rate increasing** over time (should plateau around 40-48%)
2. **Bust rate decreasing** (should stabilize around 5-10%)
3. **Average reward improving** (moving toward 0 or slightly positive)
4. **Policy loss stabilizing** (not fluctuating wildly)

## Further Improvements (Optional)

If results are still suboptimal, consider:

1. **Learning rate schedule**: Decay learning rate over time
2. **Larger network**: Try [512, 512, 256, 128]
3. **Different reward shaping**: Add intermediate rewards for good decisions
4. **Curriculum learning**: Start with easier scenarios
5. **Ensemble methods**: Train multiple agents and average predictions

