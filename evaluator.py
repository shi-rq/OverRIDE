import json
import os
from typing import Dict, Any, List, Optional
from collections import OrderedDict
from omegaconf import OmegaConf
from utils.score import default_compute_score


NAME2MODEL = {
    'Mistral7b': "mistralai/Mistral-7B-Instruct-v0.3",
    'Qwen0.5b': 'Qwen/Qwen2.5-0.5B-Instruct',
    'Qwen1.5b': 'Qwen/Qwen2.5-1.5B-Instruct',
    'Qwen3b': 'Qwen/Qwen2.5-3B-Instruct',
    'Qwen7b': 'Qwen/Qwen2.5-7B-Instruct',
    'Qwen14b': 'Qwen/Qwen2.5-14B-Instruct',
    'Qwen32b': 'Qwen/Qwen2.5-32B-Instruct',
    'Qwen72b': 'Qwen/Qwen2.5-72B-Instruct',
    'QwQ': 'Qwen/QwQ-32B',
    'Phi3.8b': 'microsoft/Phi-4-mini-instruct',
    'Llama1b': 'meta-llama/Llama3.2-1B-Instruct',
    'Llama3b': 'meta-llama/Llama3.2-3B-Instruct',
    'Llama3-8b': 'meta-llama/Meta-Llama3-8B-Instruct',
    'Llama8b': 'meta-llama/Llama3.1-8B-Instruct',
    'Gemma2b': 'google/gemma-2-2b-it',
    'Gemma4b': 'google/gemma-3-4b-it',
    'Gemma9b': 'google/gemma-2-9b-it',
    'Gemma27b': 'google/gemma-2-27b-it',

    'Minilm': 'sentence-transformers/all-MiniLM-L6-v2',
}
MODEL2NAME = {v: k for k, v in NAME2MODEL.items()}


class Evaluator:
    def __init__(self, config: Dict[str, Any], config_all: Optional[Dict[str, Any]] = None):
        """Initialize evaluator with config.
        
        Args:
            config: Configuration dictionary containing evaluator settings
            config_all: Complete configuration dictionary containing all settings
        """
        self.config = config
        self.config_all = config_all or config
        self.pass_k = config.get('pass_k', [1, 2, 5, 10, 20, 30, 50, 100])
        self.output_dir = config.get('output_dir', './results')
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
    
    def evaluate(self, dataset, responses: List[List[str]]) -> Dict[str, float]:
        """Evaluate responses using pass-k accuracy.
        
        Args:
            dataset: Dataset object containing examples
            responses: List of lists, where each inner list contains responses for one example
            
        Returns:
            Dictionary mapping pass_k to accuracy scores
        """
        # Get the number of responses per example
        n = len(responses[0]) if responses else 0
        
        # Filter pass_k values that are <= n
        valid_pass_k = [k for k in self.pass_k if k <= n]
        
        # Initialize results
        pass_k_results = OrderedDict()
        
        # Calculate pass-k accuracy for each valid k
        for k in valid_pass_k:
            correct_count = 0
            total_count = len(dataset)
            
            for i, example in enumerate(dataset):
                example_responses = responses[i][:k]  # Take first k responses
                
                # Check if any of the first k responses is correct
                is_correct = False
                for response in example_responses:
                    score = self._compute_score(example, response)
                    if score > 0:  # Assuming score > 0 means correct
                        is_correct = True
                        break
                
                if is_correct:
                    correct_count += 1
            
            accuracy = correct_count / total_count if total_count > 0 else 0.0
            pass_k_results[f"pass_{k}"] = accuracy
        
        return pass_k_results
    
    def _compute_score(self, example: Dict[str, Any], response: str) -> float:
        """Compute score for a single response using default_compute_score.
        
        Args:
            example: Dataset example containing ground truth and metadata
            response: Model response to evaluate
            
        Returns:
            Score for the response
        """
        data_source = example.get('data_source')
        ground_truth = example.get('reward_model').get('ground_truth')
        extra_info = example.get('extra_info')
        
        try:
            score = default_compute_score(
                data_source=data_source,
                solution_str=response,
                ground_truth=ground_truth,
                extra_info=extra_info
            )
            return float(score)
        except Exception as e:
            print(f"Error computing score: {e}")
            return 0.0
    
    def calculate_pass_k_score(self, dataset, responses: List[List[str]]) -> List[Dict[str, float]]:
        """Calculate pass-k score for each example.
        
        Args:
            dataset: Dataset object containing examples
            responses: List of lists, where each inner list contains responses for one example
            
        Returns:
            List of dictionaries, where each dict contains pass-k scores for one example
        """
        pass_k_scores = []
        
        for i, example in enumerate(dataset):
            example_responses = responses[i] if i < len(responses) else []
            example_scores = {}
            
            # Calculate pass-k scores for this example
            for k in self.pass_k:
                if k <= len(example_responses):
                    # Check if any of the first k responses is correct
                    is_correct = False
                    for response in example_responses[:k]:
                        score = self._compute_score(example, response)
                        if score > 0:  # Assuming score > 0 means correct
                            is_correct = True
                            break
                    
                    example_scores[f"pass_{k}"] = 1.0 if is_correct else 0.0
                else:
                    example_scores[f"pass_{k}"] = 0.0  # Not enough responses
            
            pass_k_scores.append(example_scores)
        
        return pass_k_scores
    
    def save_results(self, dataset, responses: List[List[str]], 
                    pass_k_results: Dict[str, float], 
                    pass_k_score: Optional[List[Dict[str, float]]] = None) -> str:
        """Save evaluation results to file.
        
        Args:
            dataset: Dataset object
            responses: List of response lists
            pass_k_results: Pass-k accuracy results
            pass_k_score: Optional pass-k score list (will be calculated if not provided)
            
        Returns:
            Path to the saved file
        """
        # Calculate pass_k_score if not provided
        if pass_k_score is None:
            pass_k_score = self.calculate_pass_k_score(dataset, responses)
        
        # Get dataset and model information for filename
        dataset_name = self.config_all.get('dataset', {}).get('task', 'unknown')
        model_path = self.config_all.get('engine', {}).get('model', 'unknown')
        model_name = model_path.split('/')[-1]  # Extract model name
        
        # Get engine parameters from config
        engine_config = self.config_all.get('engine', {})
        top_p = engine_config.get('top_p', 'unknown')
        top_k = engine_config.get('top_k', 'unknown')
        min_p = engine_config.get('min_p', 'unknown')
        temperature = engine_config.get('temperature', 'unknown')
        
        # Create filename
        filename_parts = [
            dataset_name,
            MODEL2NAME.get(model_path, model_name),
            f"TP{top_p}",
            f"TK{top_k}",
            f"MP{min_p}",
            f"T{temperature}"
        ]
        
        # Add pass-k accuracy to filename
        for k, acc in pass_k_results.items():
            k_value = k.split('_')[1]  # Extract number from "pass_k"
            filename_parts.append(f"{acc*100:.2f}%@{k_value}")
        
        filename = "_".join(filename_parts) + ".json"
        filepath = os.path.join(self.output_dir, filename)
        
        # Prepare data to save
        results_data = {
            'results': [],
            'pass_k_results': pass_k_results,
            'config': self.config_all
        }
        
        # Add individual example results
        for i, example in enumerate(dataset):
            # Create a copy of the example and add responses and pass_k_score
            example_with_results = example.copy()
            example_with_results['responses'] = responses[i] if i < len(responses) else []
            example_with_results['pass_k_score'] = pass_k_score[i] if i < len(pass_k_score) else {}
            results_data['results'].append(example_with_results)
        
        # Save to file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=4, ensure_ascii=False)
        
        return filepath 