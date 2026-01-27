# OverRIDE: Diverse Text Decoding via Iterative Reweighting

Official repository for the paper "Diverse Text Decoding via Iterative Reweighting" (ICLR 2026).

![Intro](Intro.png)

## 📢 Updates

We have updated the code to support vLLM v0.11.1. This version of vLLM supports recording several key metrics during offline inference. For details, please refer to [this pull request](https://github.com/vllm-project/vllm/pull/12644) and [this pull request](https://github.com/vllm-project/vllm/pull/24947). Accordingly, we have modified `gpu_model_runner.py` and `modeling.py` to adapt to the new v1 engine.

For the original implementation of OverRIDE using vLLM v0.8.5.post1, please switch to the `vllm-0.8.5.post1` branch:

```bash
git checkout vllm-0.8.5.post1
```

## 🔧 Environment

Use the following commands to create a conda environment:

```bash
conda create -n override python=3.10
conda activate override
pip install -r requirements.txt
```

## 📚 Data

Use the following commands to download the data:

```bash
python utils/data_preprocess/humaneval.py
python utils/data_preprocess/math500.py
python utils/data_preprocess/gsm8k.py
python utils/data_preprocess/ccnews.py
```

The corresponding data will be saved in the `data` folder.

## 🚀 Evaluation

Adjust the config file at `config/default.yaml` based on your device settings.

Use the following scripts to conduct corresponding experiments:

```bash
bash scripts/override.sh
bash scripts/rank.sh
bash scripts/throughput.sh
```

## 📝 File summary

- `main.py`: Main function for running the baseline / OverRIDE evaluation.
- `engine.py`: Engine class for generating formatted responses using vLLM.
- `dataset.py`: Dataset class for loading and preprocessing data.
- `evaluator.py`: Evaluator class for evaluating PASS@K scores and saving the results.
- `modeling.py`: OverRIDE model class with reweighting heads, *overriden* from vLLM's `LogitsProcessor`.
- `gpu_model_runner.py`: GPU model runner class that supports OverRIDE, *overriden* from vLLM's `GPUModelRunner`.
- `utils/`: Utility functions for preprocessing data and scoring responses. These files are modified from [verl](https://github.com/volcengine/verl).

> ⚠️ All modifications are based on the [vLLM v1 engine](https://docs.vllm.ai/en/stable/usage/v1_guide.html). You may encounter issues when implementing on specific models (e.g., Gemma2, whose `lm_head` implementation is different) or decoding methods (e.g., speculative decoding).