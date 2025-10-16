import json
import os
import time

from typing import Dict, Any, List, Optional
from collections import OrderedDict
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
    'Llama1b': 'meta-llama/Llama-3.2-1B-Instruct',
    'Llama3b': 'meta-llama/Llama-3.2-3B-Instruc t',
    'Llama3-8b': 'meta-llama/Meta-Llama3-8B-Instruct',
    'Llama8b': 'meta-llama/Llama-3.1-8B-Instruct',
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
        round_num = len(responses[0]) if responses else 0
        valid_pass_k = [k for k in self.pass_k if k <= round_num]
        pass_k_results = OrderedDict()
        round_accuracies = OrderedDict()
        scores = []

        for i, example in enumerate(dataset):
            example_responses = responses[i]
            example_scores = []
            for response in example_responses:
                score = self._compute_score(example, response)
                example_scores.append(score)
            scores.append(example_scores)
        
        # Calculate pass-k accuracy for each valid k
        for k in valid_pass_k:
            correct_count = 0
            for i, example_scores in enumerate(scores):
                if k <= len(example_scores):
                    correct_count += sum(example_scores[:k]) > 0
            accuracy = correct_count / len(scores) if len(scores) > 0 else 0.0
            pass_k_results[f"pass_{k}"] = accuracy
        
        # Calculate round accuracies
        for round_idx in range(round_num):
            round_scores = [example_scores[round_idx] for example_scores in scores]
            round_accuracies[f"round_{round_idx}"] = sum(round_scores) / len(round_scores)
        
        return {
            'scores': scores,
            'pass_k_results': pass_k_results,
            'round_accuracies': round_accuracies,
        }
    
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
    
    def save_results(self, dataset, responses: List[List[str]], results: Dict[str, Any]) -> str:
        """Save evaluation results to file.
        
        Args:
            dataset: Dataset object
            responses: List of response lists
            results: Dictionary containing evaluation results
            
        Returns:
            Path to the saved file
        """
        
        # Get dataset and model information for filename
        dataset_name = self.config_all.get('dataset', {}).get('task', 'unknown')
        model_path = self.config_all.get('engine', {}).get('model', 'unknown')
        model_name = model_path.split('/')[-1]  # Extract model name

        # Get OverRIDE parameters from config
        override_params = self.config_all.get('engine', {}).get('override', {})
        lambd = override_params.get('lambd', 'unknown')
        learning_rate = override_params.get('learning_rate', 'unknown')
        rank = override_params.get('rank', 'unknown')
        
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

        # Add override parameters to filename
        if self.config_all['main']['method'] == 'override':
            filename_parts.insert(2, f"LMD{lambd}")
            filename_parts.insert(3, f"LR{learning_rate}")
            filename_parts.insert(4, f"R{rank}")
        
        # Add pass-k accuracy to filename
        if dataset_name in ['ccnews']:
            filename_parts.append(time.strftime("%Y%m%d_%H%M%S"))
        else:
            for k, acc in results['pass_k_results'].items():
                k_value = k.split('_')[1]  # Extract number from "pass_k"
                filename_parts.append(f"{acc*100:.2f}%@{k_value}")
        
        filename = "_".join(filename_parts) + ".json"
        filepath = os.path.join(self.output_dir, filename)
        
        # Prepare data to save
        results_data = {
            'results': [],
            'config': self.config_all,
            'pass_k_results': results['pass_k_results'],
            'round_accuracies': results['round_accuracies'],
        }
        
        # Add individual example results
        for i, example in enumerate(dataset):
            # Create a copy of the example and add responses and pass_k_score
            example_with_results = example.copy()
            example_with_results['responses'] = responses[i] if i < len(responses) else []
            example_with_results['scores'] = results['scores'][i] if i < len(results['scores']) else []
            results_data['results'].append(example_with_results)
        
        # Save to file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=4, ensure_ascii=False)
        
        return filepath 