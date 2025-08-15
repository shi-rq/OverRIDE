import torch

from typing import Dict, Any, List, Optional
from vllm import LLM, SamplingParams
from tqdm import tqdm


class Engine:
    def __init__(self, config: Dict[str, Any]):
        """Initialize vLLM engine with config.
        
        Args:
            config: Configuration dictionary containing engine settings
        """
        self.config = config
        
        # Initialize vLLM engine
        self.llm = LLM(
            model=config.get('model', 'Qwen/Qwen2.5-3B-Instruct'),
            trust_remote_code=True,
            gpu_memory_utilization=0.8,
        )
        
        # Initialize sampling parameters
        self.sampling_params = SamplingParams(
            n=config.get('n', 1),
            top_p=config.get('top_p', 1.0),
            top_k=config.get('top_k', -1),
            min_p=config.get('min_p', 0.0),
            temperature=config.get('temperature', 0.0),
            max_tokens=config.get('max_tokens', 8192),
        )
        self.sampling_params_sequential = SamplingParams(
            top_p=config.get('top_p', 1.0),
            top_k=config.get('top_k', -1),
            min_p=config.get('min_p', 0.0),
            temperature=config.get('temperature', 0.0),
            max_tokens=config.get('max_tokens', 8192),
        )
    
    def format_prompts(self, messages: List[Dict[str, str]]) -> List[str]:
        """Format messages into prompts using chat template.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            
        Returns:
            List of formatted prompts
        """
        tokenizer = self.llm.get_tokenizer()
        
        prompts = []
        for message_list in messages:
            # Determine if we need to add generation prompt
            last_role = message_list[-1]['role'] if message_list else 'user'
            
            if last_role == 'user':
                add_generation_prompt = True
                continue_final_message = False
            elif last_role == 'assistant':
                add_generation_prompt = False
                continue_final_message = True
            else:
                raise ValueError(f"Invalid role for the last message: {last_role}")
            
            # Apply chat template
            prompt = tokenizer.apply_chat_template(
                message_list,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                continue_final_message=continue_final_message
            )
            prompts.append(prompt)
        
        return prompts
    
    def generate_responses(self, prompts: List[str], n: Optional[int] = None) -> List[List[str]]:
        """Generate responses for given prompts.
        
        Args:
            prompts: List of input prompts
            n: Number of responses per prompt (overrides config if provided)
            mode: Generation mode, either 'baseline' or 'sequential'

        Returns:
            List of lists, where each inner list contains n responses for the corresponding prompt
        """
        # Create sampling params for this generation
        if n is not None:
            sampling_params = SamplingParams(
                n=n,
                top_p=self.sampling_params.top_p,
                top_k=self.sampling_params.top_k,
                min_p=self.sampling_params.min_p,
                temperature=self.sampling_params.temperature,
                max_tokens=self.sampling_params.max_tokens,
            )
        else:
            sampling_params = self.sampling_params
        
        # Generate responses
        outputs = self.llm.generate(prompts, sampling_params)
        
        # Extract responses
        responses = []
        for output in outputs:
            prompt_responses = []
            for sample in output.outputs:
                prompt_responses.append(sample.text)
            responses.append(prompt_responses)
        
        return responses
    
    def generate_responses_sequential(self, prompts: List[str], n: Optional[int] = None) -> List[List[str]]:
        """Generate responses for given prompts sequentially.
        
        Args:
            prompts: List of input prompts
            n: Number of responses per prompt (overrides config if provided)
        
        Returns:
            List of lists, where each inner list contains n responses for the corresponding prompt
        """
        if n is None:
            n = self.config.get('n', 1)
        
        responses = [[] for _ in range(len(prompts))]
        
        for _ in tqdm(range(n), desc="Generating responses sequentially"):
            outputs = self.llm.generate(prompts, self.sampling_params_sequential)
            for i, output in enumerate(outputs):
                responses[i].append(output.outputs[0].text)
        
        return responses