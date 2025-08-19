#!/bin/bash
clear
set -x

# Raw evaluation script using main.py with evaluate config

# Config
MODELS=(
    "Qwen/Qwen2.5-3B-Instruct"
    "Qwen/Qwen2.5-7B-Instruct"
)
TASKS=(
    "humaneval"
    # "math"
    "gsm8k"
)
LAMBDAS=(1.0 5.0 10.0 50.0 100.0 500.0 1000.0)
LRS=(1e-6 5e-5 1e-3 5e-2)
# LAMBDAS=(2.0 0.6 1.0 1.5 0.4 0.8 1.2 0.2)
# LRS=(1e-6 5e-6 1e-5 5e-5 1e-4 5e-4 1e-3 5e-3 1e-2)

GPU_ID=0,1,2,3


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
                        engine.temperature=0.6 \
                        engine.override.lambd=$LAMBDA \
                        engine.override.learning_rate=$LR \
                        evaluator.output_dir="./results/ablation"
            done
        done
    done
done