"""
Preprocess the MATH-lighteval dataset to json format
"""

import argparse
import json
import os
import sys

import datasets

# Add parent directory to path to import hdfs_io and score.math
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hdfs_io import copy, makedirs
from score.math import last_boxed_only_string, remove_boxed


def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="./data/math")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    # 'lighteval/MATH' is no longer available on huggingface.
    # Use mirror repo: DigitalLearningGmbH/MATH-lighteval
    data_source = "DigitalLearningGmbH/MATH-lighteval"
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = datasets.load_dataset(data_source, trust_remote_code=True)

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question = example.pop("problem")

            question = question + " " + instruction_following

            answer = example.pop("solution")
            solution = extract_solution(answer)
            data = {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": question}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {"split": split, "index": idx},
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    # Create local directory if it doesn't exist
    os.makedirs(local_dir, exist_ok=True)

    # Convert to dict format with index as key
    train_dict = {item["extra_info"]["index"]: item for item in train_dataset}
    test_dict = {item["extra_info"]["index"]: item for item in test_dataset}
    
    with open(os.path.join(local_dir, "train.json"), "w") as f:
        json.dump(train_dict, f, indent=4)
    with open(os.path.join(local_dir, "validation.json"), "w") as f:
        json.dump(test_dict, f, indent=4)

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
