#!/bin/bash
clear
set -x

# Raw evaluation script using main.py with evaluate config

# Config
MODELS=(
    # "meta-llama/Llama-3.2-3B-Instruct"
    # "Qwen/Qwen2.5-3B-Instruct"
    "Qwen/Qwen2.5-7B-Instruct"
    "mistralai/Mistral-7B-Instruct-v0.3"
    "meta-llama/Llama-3.1-8B-Instruct"
    "Qwen/Qwen2.5-14B-Instruct"
    # "Qwen/Qwen2.5-32B-Instruct"
    # "Qwen/Qwen2.5-72B-Instruct"
    # "google/gemma-3-4b-it"
    # "microsoft/Phi-4-mini-instruct"
    # "google/gemma-2-9b"
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

LAMBDA=0.8
LR=1e-3

GPU_ID=0


for MODEL in "${MODELS[@]}"; do
    for TASK in "${TASKS[@]}"; do
            echo "Running Topp=$TOPP evaluation with T=0: Model=$MODEL, Task=$TASK"
            USE_OVERRIDE=true CUDA_VISIBLE_DEVICES=$GPU_ID python main.py \
                --config-path=config \
                --config-name=default \
                main.method=override \
                engine.model=$MODEL \
                dataset.task=$TASK \
                engine.n=$SAMPLE_NUM \
                engine.top_p=$TOPP \
                engine.temperature=0 \
                engine.override.lambd=$LAMBDA \
                engine.override.learning_rate=$LR \
                evaluator.output_dir="./results/override"
            
            echo "Running Topp=$TOPP evaluation with T=0.6: Model=$MODEL, Task=$TASK"
            USE_OVERRIDE=true CUDA_VISIBLE_DEVICES=$GPU_ID python main.py \
                --config-path=config \
                --config-name=default \
                main.method=override \
                engine.model=$MODEL \
                dataset.task=$TASK \
                engine.n=$SAMPLE_NUM \
                engine.top_p=$TOPP \
                engine.temperature=0.6 \
                engine.override.lambd=$LAMBDA \
                engine.override.learning_rate=$LR \
                evaluator.output_dir="./results/override"

            echo "Running Topp=$TOPP evaluation with T=1.0: Model=$MODEL, Task=$TASK"
            USE_OVERRIDE=true CUDA_VISIBLE_DEVICES=$GPU_ID python main.py \
                --config-path=config \
                --config-name=default \
                main.method=override \
                engine.model=$MODEL \
                dataset.task=$TASK \
                engine.n=$SAMPLE_NUM \
                engine.top_p=$TOPP \
                engine.temperature=1.0 \
                engine.override.lambd=$LAMBDA \
                engine.override.learning_rate=$LR \
                evaluator.output_dir="./results/override"

            echo "Running Topp=$TOPP evaluation with T=1.5: Model=$MODEL, Task=$TASK"
            USE_OVERRIDE=true CUDA_VISIBLE_DEVICES=$GPU_ID python main.py \
                --config-path=config \
                --config-name=default \
                main.method=override \
                engine.model=$MODEL \
                dataset.task=$TASK \
                engine.n=$SAMPLE_NUM \
                engine.top_p=$TOPP \
                engine.temperature=1.0 \
                engine.override.lambd=$LAMBDA \
                engine.override.learning_rate=$LR \
                evaluator.output_dir="./results/override"

            echo "Running Topk=$TOPK evaluation with T=0.6: Model=$MODEL, Task=$TASK"
            USE_OVERRIDE=true CUDA_VISIBLE_DEVICES=$GPU_ID python main.py \
                --config-path=config \
                --config-name=default \
                main.method=override \
                engine.model=$MODEL \
                dataset.task=$TASK \
                engine.n=$SAMPLE_NUM \
                engine.top_k=$TOPK \
                engine.temperature=0.6 \
                engine.override.lambd=$LAMBDA \
                engine.override.learning_rate=$LR \
                evaluator.output_dir="./results/override"
            
            echo "Running Minp=$MINP evaluation with T=0.6: Model=$MODEL, Task=$TASK"
            USE_OVERRIDE=true CUDA_VISIBLE_DEVICES=$GPU_ID python main.py \
                --config-path=config \
                --config-name=default \
                main.method=override \
                engine.model=$MODEL \
                dataset.task=$TASK \
                engine.n=$SAMPLE_NUM \
                engine.min_p=$MINP \
                engine.temperature=0.6 \
                engine.override.lambd=$LAMBDA \
                engine.override.learning_rate=$LR \
                evaluator.output_dir="./results/override"
    done
done