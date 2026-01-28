# Copyright 2026 Ruiqi Shi

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import Dict, Any, List, Optional
from vllm import LLM, SamplingParams

from modeling import OverRIDEParams

# Allow insecure serialization for V1 engine's collective_rpc
# This is needed to pass functions via collective_rpc in V1
os.environ['VLLM_ALLOW_INSECURE_SERIALIZATION'] = '1'


class Engine:
    def __init__(self, config: Dict[str, Any]):
        """Initialize vLLM engine with config.
        
        Args:
            config: Configuration dictionary containing engine settings
        """
        self.config = config
        self.config_override = config.get('override', {})

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

        # Initialize reweighting parameters
        self.reweighting_params = OverRIDEParams(
            lambd=self.config_override.get('lambd', 0.8),
            num_iteration=self.config_override.get('num_iteration', 10),
            rank=self.config_override.get('rank', 1024),
            learning_rate=self.config_override.get('learning_rate', 1e-6),
        )

        # Initialize vLLM engine
        self.llm = LLM(
            model=config.get('model', 'Qwen/Qwen2.5-3B-Instruct'),
            gpu_memory_utilization=config.get('gpu_memory_utilization', 0.6),
            max_model_len=config.get('max_model_len', 8192),
            trust_remote_code=True,
            tensor_parallel_size=config.get('tensor_parallel_size', 1),
        )
        if os.getenv('USE_OVERRIDE', 'false').lower() == 'true':
            self.set_reweighting_params_rpc(self.reweighting_params)
    
    def format_prompts(self, messages) -> List[str]:
        """Format messages into prompts using chat template.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            
        Returns:
            List of formatted prompts
        """
        # If not in chat template format, directly use the string list
        if not isinstance(messages[0], list):
            print(f"Messages are not in chat template format, directly use the string list")
            return messages

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
    
    def set_reweighting_params_rpc(self, reweighting_params: OverRIDEParams):
        def set_reweighting_params(executor):
            print(executor.model_runner.model.__class__)  # print name of the model
            # check if there are overridden functions
            print(f"has update_reweighting_head: {hasattr(executor.model_runner.model.logits_processor, 'update_reweighting_head')}")
            print(f"has set_reweighting_head: {hasattr(executor.model_runner.model.logits_processor, 'set_reweighting_head')}")
            executor.model_runner.model.logits_processor.set_reweighting_params(reweighting_params)
        self.llm.collective_rpc(set_reweighting_params)
    
    def update_reweighting_head_rpc(self):
        def update_reweighting_head(executor):
            executor.model_runner.model.logits_processor.update_reweighting_head()
        self.llm.collective_rpc(update_reweighting_head)
    
    def set_reweighting_head_rpc(self, iteration: int):
        def set_reweighting_head(executor):
            executor.model_runner.model.logits_processor.set_reweighting_head(iteration)
        self.llm.collective_rpc(set_reweighting_head)
    
    def reset_training_rpc(self):
        def reset_training(executor):
            return executor.model_runner.model.logits_processor.reset_training()
        return self.llm.collective_rpc(reset_training)[0]
    
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
    
    def generate_responses_override(self, prompts: List[str], n: Optional[int] = None) -> List[List[str]]:
        """Generate responses for given prompts with override.
        
        Args:
            prompts: List of input prompts
            n: Number of responses per prompt (overrides config if provided)
        
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