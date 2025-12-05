#!/bin/bash

# Script to delete all checkpoint .pth files while keeping best_model.pth
# Usage: ./cleanup_checkpoints.sh [dry-run]

DRY_RUN="${1:-}"

if [ "$DRY_RUN" == "--dry-run" ] || [ "$DRY_RUN" == "-n" ]; then
    echo "=========================================="
    echo "DRY RUN MODE - No files will be deleted"
    echo "=========================================="
    echo ""
    DRY_RUN_MODE=true
else
    echo "=========================================="
    echo "Cleaning up checkpoint files"
    echo "=========================================="
    echo "This will DELETE all checkpoint_*.pth files"
    echo "Keeping: best_model.pth, final_model.pth"
    echo ""
    read -p "Are you sure? (yes/no): " confirm
    if [ "$confirm" != "yes" ]; then
        echo "Aborted."
        exit 1
    fi
    DRY_RUN_MODE=false
fi

# Find all results directories
RESULTS_DIRS=(
    "results"
    "results_expected_return"
    "results_sharpe_ratio"
    "results_win_rate"
)

total_deleted=0
total_size_freed=0

for results_dir in "${RESULTS_DIRS[@]}"; do
    if [ ! -d "$results_dir" ]; then
        continue
    fi
    
    echo "Processing: $results_dir"
    
    # Find all checkpoint files in subdirectories
    while IFS= read -r -d '' checkpoint_file; do
        file_size=$(stat -f%z "$checkpoint_file" 2>/dev/null || stat -c%s "$checkpoint_file" 2>/dev/null || echo 0)
        file_size_mb=$(echo "scale=2; $file_size / 1024 / 1024" | bc -l)
        
        if [ "$DRY_RUN_MODE" = true ]; then
            echo "  Would delete: $checkpoint_file (${file_size_mb} MB)"
        else
            rm -f "$checkpoint_file"
            if [ $? -eq 0 ]; then
                echo "  ✓ Deleted: $checkpoint_file (${file_size_mb} MB)"
                total_deleted=$((total_deleted + 1))
                total_size_freed=$(echo "$total_size_freed + $file_size" | bc -l)
            else
                echo "  ✗ Failed to delete: $checkpoint_file"
            fi
        fi
    done < <(find "$results_dir" -type f -name "checkpoint_*.pth" -print0 2>/dev/null)
    
    echo ""
done

if [ "$DRY_RUN_MODE" = true ]; then
    echo "=========================================="
    echo "DRY RUN COMPLETE"
    echo "=========================================="
    echo "Run without --dry-run to actually delete files"
else
    total_size_freed_mb=$(echo "scale=2; $total_size_freed / 1024 / 1024" | bc -l)
    total_size_freed_gb=$(echo "scale=2; $total_size_freed / 1024 / 1024 / 1024" | bc -l)
    
    echo "=========================================="
    echo "Cleanup Complete!"
    echo "=========================================="
    echo "Files deleted: $total_deleted"
    echo "Space freed: ${total_size_freed_mb} MB (${total_size_freed_gb} GB)"
    echo ""
    echo "Kept files:"
    echo "  - best_model.pth (in each seed directory)"
    echo "  - final_model.pth (in each seed directory)"
    echo "  - All other non-checkpoint files"
fi

echo ""
echo "=========================================="