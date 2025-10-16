#!/bin/bash
clear
set -x

MODELS=(
    "Qwen/Qwen2.5-7B-Instruct"
    "mistralai/Mistral-7B-Instruct-v0.3"
)
TASKS=(
    "humaneval"
    "math"
    "gsm8k"
    "ccnews"
)

SAMPLE_NUM=10
TOPP=0.9
TOPK=20
MINP=0.05

if [[ "${MODELS[0]}" == *"mistral"* || "${MODELS[0]}" == *"llama"* ]]; then
    LAMBDA=0.4
else
    LAMBDA=0.8
fi
LR=1e-3

GPU_ID=0


for MODEL in "${MODELS[@]}"; do
    for TASK in "${TASKS[@]}"; do
        # Set config based on task
        if [ "$TASK" = "ccnews" ]; then
            CONFIG="story_generation"
        else
            CONFIG="default"
        fi
        
        echo "Running Topp=$TOPP evaluation with T=0.6: Model=$MODEL, Task=$TASK, Config=$CONFIG"
        USE_OVERRIDE=true CUDA_VISIBLE_DEVICES=$GPU_ID python main.py \
            --config-path=config \
            --config-name=$CONFIG \
            main.method=override \
            engine.model=$MODEL \
            dataset.task=$TASK \
            engine.n=$SAMPLE_NUM \
            engine.top_p=$TOPP \
            engine.temperature=0.6 \
            engine.override.lambd=$LAMBDA \
            engine.override.learning_rate=$LR \
            evaluator.output_dir="./results/override"

        echo "Running Topp=$TOPP evaluation with T=1.0: Model=$MODEL, Task=$TASK, Config=$CONFIG"
        USE_OVERRIDE=true CUDA_VISIBLE_DEVICES=$GPU_ID python main.py \
            --config-path=config \
            --config-name=$CONFIG \
            main.method=override \
            engine.model=$MODEL \
            dataset.task=$TASK \
            engine.n=$SAMPLE_NUM \
            engine.top_p=$TOPP \
            engine.temperature=1.0 \
            engine.override.lambd=$LAMBDA \
            engine.override.learning_rate=$LR \
            evaluator.output_dir="./results/override"

        echo "Running Topk=$TOPK evaluation with T=0.6: Model=$MODEL, Task=$TASK, Config=$CONFIG"
        USE_OVERRIDE=true CUDA_VISIBLE_DEVICES=$GPU_ID python main.py \
            --config-path=config \
            --config-name=$CONFIG \
            main.method=override \
            engine.model=$MODEL \
            dataset.task=$TASK \
            engine.n=$SAMPLE_NUM \
            engine.top_k=$TOPK \
            engine.temperature=0.6 \
            engine.override.lambd=$LAMBDA \
            engine.override.learning_rate=$LR \
            evaluator.output_dir="./results/override"
        
        echo "Running Minp=$MINP evaluation with T=0.6: Model=$MODEL, Task=$TASK, Config=$CONFIG"
        USE_OVERRIDE=true CUDA_VISIBLE_DEVICES=$GPU_ID python main.py \
            --config-path=config \
            --config-name=$CONFIG \
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