#!/bin/bash

# GPUs to use (each GPU runs only one task)
gpus=(1 2 3)
# Array to store the PID of currently running processes for each GPU
declare -A gpu_jobs

# Function to check if a GPU slot is available
wait_for_gpu() {
    while true; do
        for gpu in "${gpus[@]}"; do
            # If there is no task on the GPU, return immediately
            if [[ -z "${gpu_jobs[$gpu]}" ]]; then
                return 0
            else
                # If the process has ended, clear the slot and return
                if ! kill -0 "${gpu_jobs[$gpu]}" 2>/dev/null; then
                    gpu_jobs[$gpu]=""
                    return 0
                fi
            fi
        done
        sleep 1
    done
}

mkdir -p logs/training_ATV

weight_decay=1e-5
# Task scheduling: run for each seed, weight_fv, and dataset combination
for lr in 5e-4; do
    for epochs in 15; do
        for seed in 42 100 10; do
            for weight_fv in 0.001; do
                # Wait for an available GPU slot
                wait_for_gpu
                # Find an available GPU id
                for gpu in "${gpus[@]}"; do
                        if [[ -z "${gpu_jobs[$gpu]}" ]]; then
                            free_gpu=$gpu
                            break
                        fi
                done

                # Log file name (create a separate log file for each task)
                log_file="logs/training_ATV/log_${weight_fv}_${seed}_${weight_decay}.log"

                echo "Running weight_fv: ${weight_fv}, seed: ${seed}, weight_decay: ${weight_decay}"
                
                # Actual command configuration (original cmd variable content)
                cmd="python ATV_training.py \
                --model_name meta-llama/Meta-Llama-3-8B \
                --test_samples 90 \
                --save_dir eval_results/training_ATV/adapICV_top1_${weight_fv}_${seed}_${weight_decay} \
                --weight_ori 1.0 \
                --weight_fv ${weight_fv} \
                --dataset_split test \
                --local \
                --prompt_file dataset_files/natural_prompts_manual.json \
                --seed ${seed} \
                --soft \
                --recollect_state \
                --use_template \
                --epochs ${epochs} \
                --learning_rate ${lr} \
                --weight_decay ${weight_decay} "
                
                # Assign the allocated GPU number to an environment variable and run in the background,
                # Standard output and standard error are saved to the log file (no console output)
                CUDA_VISIBLE_DEVICES=${free_gpu} bash -c "$cmd" > "$log_file" 2>&1 &
                
                # Save the background process PID
                gpu_jobs[$free_gpu]=$!
            done
        done
    done
done

# Wait for all background tasks to complete
wait

# Final completion log (if desired, redirect to a file or remove)
echo "All tasks completed"
