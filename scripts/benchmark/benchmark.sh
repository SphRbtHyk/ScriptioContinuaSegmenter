#!/bin/bash

# Maximum number of parallel jobs
MAX_JOBS=8
current_jobs=0

# Create log directory
LOG_DIR="benchmark_logs"
mkdir -p "$LOG_DIR"

# Create a function to run your benchmark
run_benchmark() {
    local lang=$1
    local algo=$2
    local ann=$3
    local beam=$4
    local log_file="$LOG_DIR/${lang}_${algo}_${ann}_${beam}.log"
    
    echo "Starting: $lang $algo $ann $beam -> $log_file"
    python compute_scores.py --language "$lang" --algorithm "$algo" --annotation "$ann" --beam_width "$beam" > "$log_file" 2>&1
    echo "Completed: $lang $algo $ann $beam"
}

# Export the function so it's available in subshells
export -f run_benchmark

# Function to wait when too many jobs are running
wait_for_jobs() {
    while [ $(jobs -r | wc -l) -ge $MAX_JOBS ]; do
        sleep 1
    done
}

# Run all combinations in parallel (STANDARD and HIERARCHICAL only)
# languages=("seals")
languages=("grc" "lat") # Seals are absent due to copyrights
algorithms=("STANDARD" "HIERARCHICAL")
annotations=("binary" "quadri")


echo "=== Running STANDARD and HIERARCHICAL algorithms ==="
echo "Logs will be saved to: $LOG_DIR/"
for lang in "${languages[@]}"; do
    for algo in "${algorithms[@]}"; do
        for ann in "${annotations[@]}"; do
            if [ "$algo" = "STANDARD" ]; then
                wait_for_jobs
                run_benchmark "$lang" "$algo" "$ann" 1 &
                current_jobs=$((current_jobs + 1))
            elif [ "$algo" = "HIERARCHICAL" ]; then
                # for beam in 3 10 15; do
                for beam in 15; do
                    wait_for_jobs
                    run_benchmark "$lang" "$algo" "$ann" "$beam" &
                    current_jobs=$((current_jobs + 1))
                done
            fi
        done
    done
done

# Wait for STANDARD and HIERARCHICAL to complete
echo "Waiting for $current_jobs STANDARD and HIERARCHICAL jobs to complete..."
wait
echo "STANDARD and HIERARCHICAL completed!"

# Reset counter for Bayesian
current_jobs=0

languages=("lat" "grc") # Seals are absent due to copyrights

Now run BAYESIAN separately
echo "=== Running BAYESIAN algorithm ==="
for lang in "${languages[@]}"; do
    wait_for_jobs
    run_benchmark "$lang" "BAYESIAN" "binary" 1 &
    current_jobs=$((current_jobs + 1))
done

Wait for Bayesian to complete
echo "Waiting for $current_jobs BAYESIAN jobs to complete..."
wait
echo "All jobs completed!"