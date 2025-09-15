#!/bin/bash
clear
set -x

# Raw evaluation script using main.py with evaluate config

# Config
MODELS=(
    "Qwen/Qwen2.5-7B-Instruct"
    "Qwen/Qwen2.5-3B-Instruct"
    "meta-llama/Llama-3.1-8B-Instruct"
    "mistralai/Mistral-7B-Instruct-v0.3"
    # "Qwen/Qwen2.5-14B-Instruct"
    # "Qwen/Qwen2.5-32B-Instruct"
    # "Qwen/Qwen2.5-72B-Instruct"
)
TASKS=(
    "humaneval"
    # "math"
    # "gsm8k"
)

LAMBDAS=(0.1 0.3 1.0 2.0 3.0 5.0 10.0)
LRS=(1e-3 3e-5)

GPU_ID=3


for MODEL in "${MODELS[@]}"; do
    for TASK in "${TASKS[@]}"; do
        for LAMBDA in "${LAMBDAS[@]}"; do
            for LR in "${LRS[@]}"; do
                echo "Running Topp=0.9 evaluation with T=0.6: Model=$MODEL, Task=$TASK, Lambda=$LAMBDA, LR=$LR"
                USE_OVERRIDE=true CUDA_VISIBLE_DEVICES=$GPU_ID python main.py \
                    --config-path=config \
                    --config-name=default \
                    main.method=override \
                    engine.model=$MODEL \
                    dataset.task=$TASK \
                    engine.n=10 \
                    engine.top_p=0.9 \
                    engine.temperature=0.0001 \
                    engine.override.lambd=$LAMBDA \
                    engine.override.learning_rate=$LR \
                    evaluator.output_dir="./results/ablation-greedy" \
                    engine.tensor_parallel_size=1
            done
        done
    done
done