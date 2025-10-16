"""
Preprocess the GSM8k dataset to json format
"""

import argparse
import json
import os
import re
import sys

import datasets

# Add parent directory to path to import hdfs_io
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hdfs_io import copy, makedirs


def extract_solution(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split("#### ")[1].replace(",", "")
    return final_solution


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="./data/gsm8k")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    data_source = "openai/gsm8k"

    dataset = datasets.load_dataset(data_source, "main")

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    # instruction_following = 'Let\'s think step by step and output the final answer after "####".'
    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop("question")

            question = question_raw + " " + instruction_following

            answer_raw = example.pop("answer")
            solution = extract_solution(answer_raw)
            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer_raw,
                    "question": question_raw,
                },
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
    with open(os.path.join(local_dir, "test.json"), "w") as f:
        json.dump(test_dict, f, indent=4)

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
