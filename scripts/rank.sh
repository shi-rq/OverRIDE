#!/bin/bash
clear
set -x

MODELS=(
    "Qwen/Qwen2.5-7B-Instruct"
)
TASKS=(
    "math"
)

SAMPLE_NUM=10
TOPP=0.9
TOPK=20
MINP=0.05

LAMBDA=0.8
LRS=(1e-3 1e-3 1e-3 3e-4 3e-4 1e-4 1e-4 1e-4)
RANKS=(4 8 16 32 64 128 256 512)

GPU_ID=0


for MODEL in "${MODELS[@]}"; do
    for TASK in "${TASKS[@]}"; do
        for idx in "${!LRS[@]}"; do
            LR=${LRS[$idx]}
            RANK=${RANKS[$idx]}
            echo "Running OverRIDE Topp=$TOPP evaluation with T=0.6: Model=$MODEL, Task=$TASK, LR=$LR, RANK=$RANK"
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
                evaluator.output_dir="./results/rank"
        done
    done
done