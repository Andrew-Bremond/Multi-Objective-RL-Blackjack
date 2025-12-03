#!/bin/bash

# Script to train all RL algorithms for Blackjack
# Usage: ./train_all.sh

CONFIG="configs/config.yaml"
ALGORITHMS=("dqn" "ddqn" "dueling_ddqn" "ppo")
LOG_FILE="training_log_$(date +%Y%m%d_%H%M%S).txt"

# Function to convert to uppercase (compatible with bash 3.2)
to_upper() {
    echo "$1" | tr '[:lower:]' '[:upper:]'
}

echo "=========================================="
echo "Training All Algorithms"
echo "Started at: $(date)"
echo "=========================================="
echo ""

# Redirect output to log file and also display
exec > >(tee -a "$LOG_FILE")
exec 2>&1

for algo in "${ALGORITHMS[@]}"; do
    ALGO_UPPER=$(to_upper "$algo")
    echo "----------------------------------------"
    echo "Training $ALGO_UPPER..."
    echo "Started at: $(date)"
    echo "----------------------------------------"
    
    START_TIME=$(date +%s)
    
    python src/training/train.py \
        --algorithm "$algo" \
        --config "$CONFIG"
    
    EXIT_CODE=$?
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✓ $ALGO_UPPER training completed successfully"
        echo "  Duration: ${DURATION} seconds"
    else
        echo "✗ $ALGO_UPPER training failed (exit code: $EXIT_CODE)"
    fi
    
    echo ""
done

echo "=========================================="
echo "All training completed!"
echo "Finished at: $(date)"
echo "Log saved to: $LOG_FILE"
echo "=========================================="