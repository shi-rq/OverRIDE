# OverRIDE: Diverse Text Decoding via Iterative Reweighting

The source code for Diverse Text Decoding via Iterative Reweighting (OverRIDE).

![Intro](Intro.png)

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