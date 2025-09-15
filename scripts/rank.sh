#!/bin/bash
clear
set -x

# Test throughput of the model

# Config
MODELS=(
    # "Qwen/Qwen2.5-3B-Instruct"
    "Qwen/Qwen2.5-7B-Instruct"
    # "Qwen/Qwen2.5-14B-Instruct"
    # "Qwen/Qwen2.5-32B-Instruct"
    # "Qwen/Qwen2.5-72B-Instruct"
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
RANK=1024

GPU_ID=7


for i in {1..5}; do
    for MODEL in "${MODELS[@]}"; do
        for TASK in "${TASKS[@]}"; do
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
                engine.override.rank=$RANK \
                engine.tensor_parallel_size=$(echo $GPU_ID | awk -F',' '{print NF}') \
                engine.gpu_memory_utilization=0.5 \
                evaluator.output_dir="./results/rank"
        done
    done
done