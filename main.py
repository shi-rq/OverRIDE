import hydra
import os

from omegaconf import DictConfig, OmegaConf
from typing import Dict, Any

from dataset import Dataset
from engine import Engine
from evaluator import Evaluator


@hydra.main(config_path="config", config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main function for running the baseline evaluation.
    
    Args:
        cfg: Hydra configuration object
    """
    # Convert config to dictionary
    config = OmegaConf.to_container(cfg, resolve=True)
    
    # Initialize components
    print("Initializing dataset...")
    dataset = Dataset(config['dataset'])
    
    print("Initializing engine...")
    engine = Engine(config['engine'])
    
    print("Initializing evaluator...")
    evaluator = Evaluator(config['evaluator'], config)

    # Format prompts
    print("Formatting prompts...")
    messages = dataset.get_prompt_messages()
    prompts = engine.format_prompts(messages)
    
    # Generate responses
    print("Generating responses...")
    method = config.get('main', {}).get('method', 'baseline')
    if method == 'baseline':
        responses = engine.generate_responses(prompts)
    elif method == 'baseline-sequential':
        responses = engine.generate_responses_sequential(prompts)
    else:
        raise NotImplementedError(f"Method `{method}` is not implemented.")
    
    # Evaluate responses
    print("Evaluating responses...")
    pass_k_results = evaluator.evaluate(dataset, responses)
    
    # Calculate pass-k score
    pass_k_score = evaluator.calculate_pass_k_score(dataset, responses)
    
    # Display results
    print("\n=== Evaluation Results ===")
    for k, accuracy in pass_k_results.items():
        print(f"{k}: {accuracy:.4f}")
    
    # Save results
    print("\nSaving results...")
    output_file = evaluator.save_results(dataset, responses, pass_k_results, pass_k_score)
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main() 