#!/bin/bash
clear
set -x

# Test throughput of the model

# Config
MODELS=(
    # "Qwen/Qwen2.5-3B-Instruct"
    # "Qwen/Qwen2.5-7B-Instruct"
    # "Qwen/Qwen2.5-14B-Instruct"
    # "Qwen/Qwen2.5-32B-Instruct"
    "Qwen/Qwen2.5-72B-Instruct"
)
TASKS=(
    # "humaneval"
    "math"
    # "gsm8k"
)

SAMPLE_NUM=10
TOPP=0.9
TOPK=20
MINP=0.05

LAMBDA=0.8
LR=1e-3

GPU_ID=0,1,2,3,4,5,6,7


for MODEL in "${MODELS[@]}"; do
    for TASK in "${TASKS[@]}"; do
        echo "Running Baseline Topp=$TOPP evaluation with T=0.6: Model=$MODEL, Task=$TASK"
        USE_OVERRIDE=false CUDA_VISIBLE_DEVICES=$GPU_ID python main.py \
            --config-path=config \
            --config-name=default \
            main.method=baseline \
            engine.model=$MODEL \
            dataset.task=$TASK \
            engine.n=$SAMPLE_NUM \
            engine.top_p=$TOPP \
            engine.temperature=0.6 \
            engine.tensor_parallel_size=$(echo $GPU_ID | awk -F',' '{print NF}') \
            evaluator.output_dir="./results/throughput"
        
        echo "Running OverRIDE Topp=$TOPP evaluation with T=0.6: Model=$MODEL, Task=$TASK"
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
            engine.tensor_parallel_size=$(echo $GPU_ID | awk -F',' '{print NF}') \
            evaluator.output_dir="./results/throughput"
    done
done