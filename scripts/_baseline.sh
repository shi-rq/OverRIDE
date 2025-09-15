#!/bin/bash
clear
set -x

# Raw evaluation script using main.py with evaluate config

# Config
MODELS=(
    "mistralai/Mistral-7B-Instruct-v0.3"
    "Qwen/Qwen2.5-7B-Instruct"
    "Qwen/Qwen2.5-3B-Instruct"
    "meta-llama/Llama-3.1-8B-Instruct"
    "Qwen/Qwen2.5-14B-Instruct"
)
TASKS=(
    "humaneval"
    "math"
    "gsm8k"
)

SAMPLE_NUM=10
TOPP=0.9
TOPK=20
MINP=0.05

GPU_ID=7


for MODEL in "${MODELS[@]}"; do
    for TASK in "${TASKS[@]}"; do
            echo "Running Topp=$TOPP evaluation with T=0: Model=$MODEL, Task=$TASK"
            CUDA_VISIBLE_DEVICES=$GPU_ID python main.py \
                --config-path=config \
                --config-name=default \
                engine.model=$MODEL \
                dataset.task=$TASK \
                engine.n=$SAMPLE_NUM \
                engine.top_p=$TOPP \
                engine.temperature=0 \
                evaluator.output_dir="./results/baseline"
    done
done