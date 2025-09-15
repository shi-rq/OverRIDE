#!/bin/bash
clear
set -x

# Raw evaluation script using main.py with evaluate config

# Config
MODELS=(
    # "Qwen/Qwen2.5-7B-Instruct"
    # "Qwen/Qwen2.5-3B-Instruct"
    # "meta-llama/Llama-3.1-8B-Instruct"
    # "mistralai/Mistral-7B-Instruct-v0.3"
    "Qwen/Qwen2.5-14B-Instruct"
)
TASKS=(
    "humaneval"
    "gsm8k"
    "math"
)

SAMPLE_NUM=10
TOPP=0.9
TOPK=20
MINP=0.05

GPU_ID=2,3


for i in {1..3}; do
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

                # echo "Running Topp=$TOPP evaluation with T=1.5: Model=$MODEL, Task=$TASK"
                # CUDA_VISIBLE_DEVICES=$GPU_ID python main.py \
                #     --config-path=config \
                #     --config-name=default \
                #     engine.model=$MODEL \
                #     dataset.task=$TASK \
                #     engine.n=$SAMPLE_NUM \
                #     engine.top_p=$TOPP \
                #     engine.temperature=1.0 \
                #     evaluator.output_dir="./results/baseline"

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
done