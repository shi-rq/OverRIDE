#!/bin/bash
clear
set -x

# Raw evaluation script using main.py with evaluate config

# Config
MODELS=(
    "Qwen/Qwen2.5-7B-Instruct"
    "Meta-llama/Llama-3.1-8B-Instruct"
    "Qwen/Qwen2.5-3B-Instruct"
    "Meta-llama/Llama-3.2-3B-Instruct"
    "Qwen/Qwen2.5-14B-Instruct"
    # "Qwen/Qwen2.5-32B-Instruct"
    # "Qwen/Qwen2.5-72B-Instruct"
    # "google/gemma-3-4b-it"
    # "microsoft/Phi-4-mini-instruct"
    "mistralai/Mistral-7B-Instruct-v0.3"
    # "google/gemma-2-9b"
)
TASKS=(
    # "humaneval"
    # "math"
    "gsm8k"
)

SAMPLE_NUM=10
TOPP=0.9
TOPK=20
MINP=0.05

GPU_ID=0,1,2,3


for MODEL in "${MODELS[@]}"; do
    for TASK in "${TASKS[@]}"; do
            echo "Running Topp=$TOPP evaluation with T=0.6: Model=$MODEL, Task=$TASK"
            CUDA_VISIBLE_DEVICES=$GPU_ID python main.py \
                --config-path=config \
                --config-name=default \
                engine.model=$MODEL \
                dataset.task=$TASK \
                engine.n=$SAMPLE_NUM \
                engine.top_p=$TOPP \
                engine.temperature=0.6 \
                evaluator.output_dir="./results/baseline"

            echo "Running Topp=$TOPP evaluation with T=1.0: Model=$MODEL, Task=$TASK"
            CUDA_VISIBLE_DEVICES=$GPU_ID python main.py \
                --config-path=config \
                --config-name=default \
                engine.model=$MODEL \
                dataset.task=$TASK \
                engine.n=$SAMPLE_NUM \
                engine.top_p=$TOPP \
                engine.temperature=1.0 \
                evaluator.output_dir="./results/baseline"

            echo "Running Topp=$TOPP evaluation with T=1.5: Model=$MODEL, Task=$TASK"
            CUDA_VISIBLE_DEVICES=$GPU_ID python main.py \
                --config-path=config \
                --config-name=default \
                engine.model=$MODEL \
                dataset.task=$TASK \
                engine.n=$SAMPLE_NUM \
                engine.top_p=$TOPP \
                engine.temperature=1.0 \
                evaluator.output_dir="./results/baseline"

            echo "Running Topk=$TOPK evaluation with T=0.6: Model=$MODEL, Task=$TASK"
            CUDA_VISIBLE_DEVICES=$GPU_ID python main.py \
                --config-path=config \
                --config-name=default \
                engine.model=$MODEL \
                dataset.task=$TASK \
                engine.n=$SAMPLE_NUM \
                engine.top_k=$TOPK \
                engine.temperature=0.6 \
                evaluator.output_dir="./results/baseline"
            
            echo "Running Minp=$MINP evaluation with T=0.6: Model=$MODEL, Task=$TASK"
            CUDA_VISIBLE_DEVICES=$GPU_ID python main.py \
                --config-path=config \
                --config-name=default \
                engine.model=$MODEL \
                dataset.task=$TASK \
                engine.n=$SAMPLE_NUM \
                engine.min_p=$MINP \
                engine.temperature=0.6 \
                evaluator.output_dir="./results/baseline"
    done
done