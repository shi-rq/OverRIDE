import hydra
import os

from omegaconf import DictConfig, OmegaConf
from typing import Dict, Any

# Check environment variable for OverRIDE monkey patch
if os.getenv('USE_OVERRIDE', 'false').lower() == 'true':
    import vllm.model_executor.layers.logits_processor as logits_processor_module
    from modeling import OverRIDELogitsProcessor
    logits_processor_module.LogitsProcessor = OverRIDELogitsProcessor
    import vllm.v1.worker.gpu_model_runner as gpu_model_runner_module
    from gpu_model_runner import GPUModelRunnerForOverRIDE
    gpu_model_runner_module.GPUModelRunner = GPUModelRunnerForOverRIDE


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
    from dataset import Dataset
    dataset = Dataset(config['dataset'])
    
    print("Initializing engine...")
    from engine import Engine
    engine = Engine(config['engine'])
    
    print("Initializing evaluator...")
    from evaluator import Evaluator
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
    elif method == 'override':
        responses = engine.generate_responses_override(prompts)
    elif method == 'override-sequential':
        responses = engine.generate_responses_override_sequential(prompts)
    else:
        raise NotImplementedError(f"Method `{method}` is not implemented.")
    
    # Evaluate responses
    print("Evaluating responses...")
    results = evaluator.evaluate(dataset, responses)
    
    # Display results
    print("\n=== Evaluation Results ===")
    for k, accuracy in results['pass_k_results'].items():
        print(f"{k}: {accuracy:.4f}")
    
    # Save results
    print("\nSaving results...")
    output_file = evaluator.save_results(dataset, responses, results)
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main() 