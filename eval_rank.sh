#!/bin/bash

# Script to rank all best_model.pth files from best to worst
# Usage: ./rank_models.sh

RESULTS_BASE="."
CONFIG="configs/config.yaml"
EVAL_EPISODES=1000
OUTPUT_FILE="model_rankings.txt"

echo "=========================================="
echo "Ranking All Best Models"
echo "=========================================="
echo ""
echo "Searching for best_model.pth files..."
echo "Evaluating with ${EVAL_EPISODES} episodes per model"
echo ""

# Temporary file to store results
TEMP_FILE=$(mktemp)

# Find all results directories (including results_* subdirectories)
RESULTS_DIRS=(
    "${RESULTS_BASE}/results"
    "${RESULTS_BASE}/results_expected_return"
    "${RESULTS_BASE}/results_sharpe_ratio"
    "${RESULTS_BASE}/results_win_rate"
)

model_count=0
evaluated_count=0

# Find all best_model.pth files
for results_dir in "${RESULTS_DIRS[@]}"; do
    if [ ! -d "$results_dir" ]; then
        continue
    fi
    
    # Get category name (e.g., "results", "results_expected_return")
    category=$(basename "$results_dir")
    
    # Find all algorithm subdirectories
    for algo_dir in "${results_dir}"/*/; do
        if [ ! -d "$algo_dir" ]; then
            continue
        fi
        
        # Find all seed subdirectories
        for seed_dir in "${algo_dir}"*/; do
            if [ ! -d "$seed_dir" ]; then
                continue
            fi
            
            best_model="${seed_dir}best_model.pth"
            if [ -f "$best_model" ]; then
                model_count=$((model_count + 1))
                
                # Extract algorithm name from path
                # Path format: results/dqn/dqn_5_timestamp/ or results_expected_return/ppo/ppo_10_timestamp/
                algo=$(basename "$algo_dir")
                
                # Extract seed from directory name (format: algo_seed_timestamp)
                seed_dir_name=$(basename "$seed_dir")
                seed=$(echo "$seed_dir_name" | sed -n 's/.*_\([0-9]*\)_.*/\1/p')
                if [ -z "$seed" ]; then
                    seed="unknown"
                fi
                
                # Create model identifier
                model_id="${category}/${algo}/seed_${seed}"
                
                echo "[$model_count] Evaluating: $model_id"
                echo "    Path: $best_model"
                
                # Evaluate model and extract Mean Reward
                eval_output=$(python src/training/evaluate.py \
                    --model_path "$best_model" \
                    --algorithm "$algo" \
                    --config "$CONFIG" \
                    --num_episodes "$EVAL_EPISODES" 2>&1)
                
                # Extract Mean Reward (primary ranking metric)
                mean_reward=$(echo "$eval_output" | grep "Mean Reward" | awk '{print $3}')
                
                # Also extract other metrics for the ranking file
                mean_return=$(echo "$eval_output" | grep "Mean Return" | awk '{print $3}')
                win_rate=$(echo "$eval_output" | grep "Win Rate" | awk '{print $3}' | sed 's/%//')
                sharpe_ratio=$(echo "$eval_output" | grep "Sharpe Ratio" | awk '{print $3}')
                
                if [ -n "$mean_reward" ] && [ "$mean_reward" != "failed" ]; then
                    evaluated_count=$((evaluated_count + 1))
                    
                    # Store results in temp file: mean_reward|model_id|path|mean_return|win_rate|sharpe_ratio
                    echo "${mean_reward}|${model_id}|${best_model}|${mean_return}|${win_rate}|${sharpe_ratio}" >> "$TEMP_FILE"
                    
                    echo "    ✓ Mean Reward: $mean_reward"
                else
                    echo "    ✗ Evaluation failed"
                fi
                echo ""
            fi
        done
    done
done

echo "=========================================="
echo "Found $model_count models, successfully evaluated $evaluated_count"
echo "=========================================="
echo ""

# Sort by mean_reward (descending) and create ranking file
if [ -s "$TEMP_FILE" ]; then
    echo "Creating ranking file: $OUTPUT_FILE"
    echo ""
    
    # Write header
    {
        echo "=========================================="
        echo "Model Rankings (Best to Worst)"
        echo "Ranked by Mean Reward"
        echo "=========================================="
        echo ""
        echo "Generated: $(date)"
        echo "Evaluation episodes per model: $EVAL_EPISODES"
        echo ""
        printf "%-6s %-50s %-12s %-12s %-10s %-12s %s\n" \
            "Rank" "Model" "Mean Reward" "Mean Return" "Win Rate" "Sharpe Ratio" "Path"
        echo "--------------------------------------------------------------------------------------------------------------------------------"
    } > "$OUTPUT_FILE"
    
    # Sort by mean_reward (first field, descending) and write ranked results
    rank=1
    sort -t'|' -k1 -rn "$TEMP_FILE" | while IFS='|' read -r mean_reward model_id path mean_return win_rate sharpe_ratio; do
        printf "%-6d %-50s %-12s %-12s %-10s %-12s %s\n" \
            "$rank" "$model_id" "$mean_reward" "$mean_return" "$win_rate" "$sharpe_ratio" "$path" >> "$OUTPUT_FILE"
        rank=$((rank + 1))
    done
    
    echo "" >> "$OUTPUT_FILE"
    echo "==========================================" >> "$OUTPUT_FILE"
    echo "End of Rankings" >> "$OUTPUT_FILE"
    echo "==========================================" >> "$OUTPUT_FILE"
    
    echo "✅ Rankings saved to $OUTPUT_FILE"
    echo ""
    echo "Top 10 Models:"
    echo "----------------------------------------"
    head -n 12 "$OUTPUT_FILE" | tail -n 10
else
    echo "❌ No models were successfully evaluated!"
fi

# Clean up
rm -f "$TEMP_FILE"

echo ""
echo "=========================================="
echo "✅ Ranking complete!"
echo "=========================================="