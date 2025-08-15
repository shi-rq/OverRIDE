"""
Preprocess the CCNews dataset to json format
"""

import argparse
import json
import os
import sys

import datasets
from transformers import AutoTokenizer

# Add parent directory to path to import hdfs_io and score.math
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hdfs_io import copy, makedirs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="./data/ccnews")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--prompt_token_length", type=int, default=32)
    parser.add_argument("--response_token_length", type=int, default=256)

    args = parser.parse_args()
    prompt_token_length = args.prompt_token_length
    response_token_length = args.response_token_length

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    data_source = "vblagoje/cc_news"
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = datasets.load_dataset(data_source, trust_remote_code=True)

    train_dataset = dataset["train"].select(range(5000, 10000))
    test_dataset = dataset["train"].select(range(5000))

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            for col in ['domain', 'url', 'image_url']:
                example.pop(col)
            text = example.pop("text")
            tokens = tokenizer(text)["input_ids"]
            question = tokenizer.decode(tokens[:prompt_token_length]) if len(tokens) >= prompt_token_length else text
            solution = tokenizer.decode(tokens[prompt_token_length:prompt_token_length+response_token_length]) if len(tokens) > prompt_token_length else ""

            title = example.pop("title")
            description = example.pop("description")

            data = {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": question}],
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {"split": split, "index": idx, "title": title, "description": description},
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
