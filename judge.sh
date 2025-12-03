#!/bin/bash

# Script to compare models and run bankroll simulations
# Usage: ./compare_and_simulate.sh

RESULTS_BASE="."
CONFIG="configs/config.yaml"
EVAL_EPISODES=100
BANKROLL_EPISODES=100

echo "=========================================="
echo "Comparing Models and Running Simulations"
echo "=========================================="
echo ""

# Find all results directories
for category_dir in ${RESULTS_BASE}/results; do
    if [ ! -d "$category_dir" ]; then
        continue
    fi
    
    category=$(basename "$category_dir")
    echo "----------------------------------------"
    echo "Category: $category"
    echo "----------------------------------------"
    
    # Find all best_model.pth files
    best_models=()
    for model_dir in ${category_dir}/*/; do
        best_model="${model_dir}best_model.pth"
        if [ -f "$best_model" ]; then
            # Extract algorithm name from directory - remove timestamp pattern (YYYYMMDD_HHMMSS)
            algo=$(basename "$model_dir" | sed 's/_[0-9]\{8\}_[0-9]\{6\}$//')
            best_models+=("${algo}:${best_model}")
        fi
    done
    
    if [ ${#best_models[@]} -eq 0 ]; then
        echo "  No models found"
        continue
    fi
    
    # Evaluate all models
    echo "  Evaluating models..."
    best_score=-999999
    best_model=""
    best_algo=""
    
    for model_info in "${best_models[@]}"; do
        algo=$(echo "$model_info" | cut -d':' -f1)
        model_path=$(echo "$model_info" | cut -d':' -f2-)
        
        echo -n "    $algo: "
        
        # Quick evaluation
        score=$(python src/training/evaluate.py \
            --model_path "$model_path" \
            --algorithm "$algo" \
            --config "$CONFIG" \
            --num_episodes "$EVAL_EPISODES" 2>/dev/null | \
            grep "Mean Return" | awk '{print $3}')
        
        if [ -n "$score" ]; then
            echo "$score"
            # Compare scores (assuming score is a number)
            if (( $(echo "$score > $best_score" | bc -l) )); then
                best_score=$score
                best_model=$model_path
                best_algo=$algo
            fi
        else
            echo "failed"
        fi
    done
    
    if [ -n "$best_model" ]; then
        echo "  ✅ Best: $best_algo (Score: $best_score)"
        
        # Run bankroll simulation
        output_name=$(echo "$category" | sed 's/results_//' | tr '_' '-')
        echo "  🎰 Running bankroll simulation..."
        
        python src/training/simulate_bankroll.py \
            --initial_bankroll 100 \
            --num_iterations "$BANKROLL_EPISODES" \
            --model_path "$best_model" \
            --algorithm "$best_algo" \
            --config "$CONFIG"
        
        # Rename output
        if [ -f "bankroll_simulation.png" ]; then
            mv "bankroll_simulation.png" "bankroll_simulation_${output_name}.png"
            echo "  ✅ Saved to bankroll_simulation_${output_name}.png"
        fi
    fi
    
    echo ""
done

echo "=========================================="
echo "✅ All comparisons and simulations completed!"
echo "=========================================="