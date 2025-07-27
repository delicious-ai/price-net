#!/bin/bash

set -e  # Exit on any error

# Configuration
DATASET_DIR="data/breps/test"
EVALUATION_SCRIPT="src/price_net/extraction/evaluate.py"
NUM_ITERATIONS=3

# Config files to evaluate
CONFIG_FILES=(
    "configs/extraction/gemini_2.5_flash.yaml"
    "configs/extraction/gemini_2.5_pro.yaml"
    "configs/extraction/gemini_2.0_flash.yaml"
)

echo "Starting automated evaluation of price extraction models..."
echo "Dataset directory: $DATASET_DIR"
echo "Number of iterations per config: $NUM_ITERATIONS"
echo "=================================================="

# Function to extract model name from config path
extract_model_name() {
    local config_path="$1"
    # Extract filename without extension
    basename "$config_path" .yaml
}

# Main evaluation loop
for config_file in "${CONFIG_FILES[@]}"; do
    echo ""
    echo "Processing config: $config_file"

    # Extract model name for experiment naming
    model_name=$(extract_model_name "$config_file")
    echo "Model name: $model_name"

    # Check if config file exists
    if [[ ! -f "$config_file" ]]; then
        echo "ERROR: Config file $config_file not found!"
        continue
    fi

    # Run evaluation multiple times for this config
    for iteration in $(seq 1 $NUM_ITERATIONS); do
        exp_name="${model_name}-${iteration}"
        echo "  Running iteration $iteration/$NUM_ITERATIONS (exp-name: $exp_name)"

        # Execute the evaluation script
        python "$EVALUATION_SCRIPT" \
            --config "$config_file" \
            --dataset-dir "$DATASET_DIR" \
            --exp-name "$exp_name"

        echo "  ✓ Completed iteration $iteration for $model_name"
    done

    echo "✓ Completed all iterations for $model_name"
done

echo ""
echo "=================================================="
echo "All evaluations completed successfully!"
echo "Total experiments run: $((${#CONFIG_FILES[@]} * NUM_ITERATIONS))"
