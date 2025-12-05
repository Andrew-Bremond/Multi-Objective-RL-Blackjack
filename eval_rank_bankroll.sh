#!/bin/bash

# Script to run bankroll simulations for all models in model_rankings.txt
# Usage: ./simulate_rankings.sh [model_rankings.txt]

RANKINGS_FILE="${1:-model_rankings.txt}"
CONFIG="configs/config.yaml"
NUM_ITERATIONS=100
INITIAL_BANKROLL=100.0
OUTPUT_FILE="bankroll_simulation_results.txt"

if [ ! -f "$RANKINGS_FILE" ]; then
    echo "Error: Rankings file not found: $RANKINGS_FILE"
    exit 1
fi

echo "=========================================="
echo "Running Bankroll Simulations"
echo "=========================================="
echo "Reading models from: $RANKINGS_FILE"
echo "Simulations: $NUM_ITERATIONS iterations per model"
echo "Initial Bankroll: \$$INITIAL_BANKROLL"
echo ""

# Temporary file to store results
TEMP_FILE=$(mktemp)

# Extract algorithm from path
extract_algorithm() {
    local path="$1"
    # Check for each algorithm in the path
    if [[ "$path" == *"dueling_ddqn"* ]]; then
        echo "dueling_ddqn"
    elif [[ "$path" == *"ddqn"* ]]; then
        echo "ddqn"
    elif [[ "$path" == *"dqn"* ]]; then
        echo "dqn"
    elif [[ "$path" == *"ppo"* ]]; then
        echo "ppo"
    else
        echo "unknown"
    fi
}

# Parse rankings file and extract model information
model_count=0
successful_count=0

# Skip header lines and process each model
while IFS= read -r line; do
    # Skip header lines and separators
    if [[ "$line" == *"Rank"* ]] || [[ "$line" == *"---"* ]] || [[ "$line" == *"==="* ]] || [[ -z "$line" ]]; then
        continue
    fi
    
    # Parse the line: Rank, Model, Mean Reward, Mean Return, Win Rate, Sharpe Ratio, Path
    # Format: "1      results/ddqn/seed_20251204                         -0.24        0.80         45.80      0.18         ./results/ddqn/ddqn_15_20251204_133936/best_model.pth"
    
    # Extract rank (first field)
    rank=$(echo "$line" | awk '{print $1}')
    
    # Extract path (last field)
    path=$(echo "$line" | awk '{print $NF}')
    
    # Skip if rank is not a number
    if ! [[ "$rank" =~ ^[0-9]+$ ]]; then
        continue
    fi
    
    # Extract model identifier (everything between rank and the metrics)
    # This is a bit tricky - we'll reconstruct it from the path
    model_id=$(echo "$line" | awk '{
        # Skip first field (rank), then collect fields until we hit a number that looks like a metric
        for (i=2; i<=NF; i++) {
            if ($i ~ /^-?[0-9]+\.[0-9]+$/) {
                break
            }
            if (i > 2) printf " "
            printf "%s", $i
        }
    }')
    
    # Clean up path (remove leading ./ if present)
    path="${path#./}"
    
    # Extract algorithm from path
    algorithm=$(extract_algorithm "$path")
    
    if [ "$algorithm" == "unknown" ]; then
        echo "Warning: Could not determine algorithm for path: $path"
        continue
    fi
    
    model_count=$((model_count + 1))
    
    echo "[$model_count] Simulating Rank $rank: $model_id"
    echo "    Algorithm: $algorithm"
    echo "    Path: $path"
    
    # Run simulation
    sim_output=$(python src/training/simulate_bankroll.py \
        --initial_bankroll "$INITIAL_BANKROLL" \
        --num_iterations "$NUM_ITERATIONS" \
        --model_path "$path" \
        --algorithm "$algorithm" \
        --config "$CONFIG" 2>&1)
    
    # Extract results from output
    initial_bankroll=$(echo "$sim_output" | grep "Initial Bankroll:" | awk '{print $3}' | sed 's/\$//')
    final_bankroll=$(echo "$sim_output" | grep "Final Bankroll:" | awk '{print $3}' | sed 's/\$//')
    total_change=$(echo "$sim_output" | grep "Total Change:" | awk '{print $3}' | sed 's/\$//' | sed 's/(.*//')
    percent_change=$(echo "$sim_output" | grep "Total Change:" | sed 's/.*(\([^)]*\)).*/\1/')
    win_rate=$(echo "$sim_output" | grep "Wins:" | awk '{print $2}' | sed 's/(\([^)]*\))/\1/' | sed 's/%//')
    games_played=$(echo "$sim_output" | grep "Games Played:" | awk '{print $3}')
    
    if [ -n "$initial_bankroll" ] && [ -n "$final_bankroll" ] && [ -n "$win_rate" ]; then
        successful_count=$((successful_count + 1))
        
        # Calculate change
        change=$(echo "$final_bankroll - $initial_bankroll" | bc -l)
        
        # Store results: rank|model_id|algorithm|initial|final|change|percent_change|win_rate|games_played|path
        echo "${rank}|${model_id}|${algorithm}|${initial_bankroll}|${final_bankroll}|${change}|${percent_change}|${win_rate}|${games_played}|${path}" >> "$TEMP_FILE"
        
        echo "    ✓ Initial: \$${initial_bankroll}, Final: \$${final_bankroll}, Change: ${percent_change}, Win Rate: ${win_rate}%"
    else
        echo "    ✗ Failed to extract results"
    fi
    echo ""
    
done < "$RANKINGS_FILE"

echo "=========================================="
echo "Completed: $successful_count/$model_count models simulated successfully"
echo "=========================================="
echo ""

# Create summary report
if [ -s "$TEMP_FILE" ]; then
    echo "Creating summary report: $OUTPUT_FILE"
    echo ""
    
    # Write header
    {
        echo "=========================================="
        echo "Bankroll Simulation Results"
        echo "=========================================="
        echo ""
        echo "Generated: $(date)"
        echo "Source: $RANKINGS_FILE"
        echo "Simulations: $NUM_ITERATIONS iterations per model"
        echo "Initial Bankroll: \$$INITIAL_BANKROLL"
        echo ""
        printf "%-6s %-45s %-10s %-12s %-12s %-12s %-10s %-8s %s\n" \
            "Rank" "Model" "Algorithm" "Initial $" "Final $" "Change" "Change %" "Win Rate" "Path"
        echo "------------------------------------------------------------------------------------------------------------------------------------------------------------------------"
    } > "$OUTPUT_FILE"
    
    # Sort by rank and write results
    sort -t'|' -k1 -n "$TEMP_FILE" | while IFS='|' read -r rank model_id algorithm initial final change percent_change win_rate games_played path; do
        printf "%-6s %-45s %-10s %-12.2f %-12.2f %-12.2f %-10s %-8s %s\n" \
            "$rank" "$model_id" "$algorithm" "$initial" "$final" "$change" "$percent_change" "$win_rate" "$path" >> "$OUTPUT_FILE"
    done
    
    echo "" >> "$OUTPUT_FILE"
    echo "==========================================" >> "$OUTPUT_FILE"
    echo "Summary Statistics" >> "$OUTPUT_FILE"
    echo "==========================================" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
    
    # Calculate statistics
    total_models=$(wc -l < "$TEMP_FILE")
    avg_final=$(awk -F'|' '{sum+=$5} END {print sum/NR}' "$TEMP_FILE")
    avg_change=$(awk -F'|' '{sum+=$6} END {print sum/NR}' "$TEMP_FILE")
    avg_win_rate=$(awk -F'|' '{sum+=$8} END {print sum/NR}' "$TEMP_FILE")
    best_final=$(awk -F'|' 'BEGIN{max=-999999} {if($5>max) max=$5} END {print max}' "$TEMP_FILE")
    worst_final=$(awk -F'|' 'BEGIN{min=999999} {if($5<min) min=$5} END {print min}' "$TEMP_FILE")
    
    {
        echo "Total Models Simulated: $total_models"
        echo "Average Final Bankroll: \$$(printf "%.2f" $avg_final)"
        echo "Average Change: \$$(printf "%.2f" $avg_change)"
        echo "Average Win Rate: $(printf "%.2f" $avg_win_rate)%"
        echo "Best Final Bankroll: \$$(printf "%.2f" $best_final)"
        echo "Worst Final Bankroll: \$$(printf "%.2f" $worst_final)"
        echo ""
        echo "=========================================="
        echo "End of Report"
        echo "=========================================="
    } >> "$OUTPUT_FILE"
    
    echo "✅ Results saved to $OUTPUT_FILE"
    echo ""
    echo "Top 10 by Final Bankroll:"
    echo "----------------------------------------"
    # Sort by final bankroll (descending) and show top 10
    sort -t'|' -k5 -rn "$TEMP_FILE" | head -n 10 | while IFS='|' read -r rank model_id algorithm initial final change percent_change win_rate games_played path; do
        printf "Rank %-3s: %-45s Final: \$%-10.2f Change: %-10s Win Rate: %-6s%%\n" \
            "$rank" "$model_id" "$final" "$percent_change" "$win_rate"
    done
else
    echo "❌ No models were successfully simulated!"
fi

# Clean up
rm -f "$TEMP_FILE"

echo ""
echo "=========================================="
echo "✅ Simulation complete!"
echo "=========================================="