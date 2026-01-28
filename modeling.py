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

import json
from typing import Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# LogitsProcessor
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding

@dataclass
class OverRIDEParams:
    """Parameters for OverRIDE reweighting mechanism"""
    lambd: float = 0.4
    num_iteration: int = 10
    rank: int = 16
    learning_rate: float = 1e-3


def custom_print(note, *items):
    """Print a note and one or more items if provided"""
    print('=' * 100)
    print(note)
    if items:
        print('-' * 70)
        for item in items:
            if isinstance(item, dict):
                print(json.dumps(item, indent=4))
            else:
                print(item)
    print('=' * 100)


class ReweightingHead(nn.Module):
    """A single reweighting head with LoRA-like structure"""
    
    def __init__(self, hidden_size, vocab_size, rank, device, dtype):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.rank = rank
        self.device = device
        self.dtype = dtype
        
        # LoRA-like structure: (hidden_size x rank, rank x vocab_size)
        self.reweighting_A = nn.Linear(hidden_size, rank, bias=False, dtype=dtype).to(device)
        self.reweighting_B = nn.Linear(rank, vocab_size, bias=False, dtype=dtype).to(device)
        
        self.reweighting_head_init()
    
    def reweighting_head_init(self):
        """Initialize the reweighting head parameters"""
        # Initialize A with normal distribution
        nn.init.normal_(self.reweighting_A.weight, mean=0.0, std=0.02)
        # Initialize B with zeros
        nn.init.zeros_(self.reweighting_B.weight)
    
    def forward(self, hidden_states):
        """Forward pass through the reweighting head"""
        return self.reweighting_B(self.reweighting_A(hidden_states))


class OverRIDELogitsProcessor(LogitsProcessor):
    def __init__(self,
                 vocab_size: int,
                 org_vocab_size: Optional[int] = None,
                 scale: float = 1.0,
                 logits_as_input: bool = False,
                 soft_cap: Optional[float] = None,
                 reweighting_params: OverRIDEParams=OverRIDEParams()) -> None:
        """
        Args:
            scale: A scaling factor to apply to the logits.
        """
        super().__init__(vocab_size, org_vocab_size, scale, logits_as_input, soft_cap)
        self.reweighting_params = reweighting_params
        custom_print("reweighting params init as", self.reweighting_params)
        self.original_lm_head = None
        self.reweighting_heads = None
        self.multiple_scales = None
        self.multiple_soft_caps = None
    
    def set_reweighting_params(self, reweighting_params: OverRIDEParams):
        self.reweighting_params = reweighting_params
        custom_print("reweighting params set to", self.reweighting_params)
        if self.original_lm_head is not None:
            self.set_reweighting_heads(self.original_lm_head)

    @torch.inference_mode(False)  # Disable inference mode to enable gradients
    def set_reweighting_heads(self, lm_head):
        """duplicate lm_head according to reweighting_params.num_iteration"""
        self.original_lm_head = lm_head
        self.reweighting_heads = nn.ModuleList([
            ReweightingHead(
                hidden_size=lm_head.embedding_dim,
                vocab_size=self.org_vocab_size,
                rank=self.reweighting_params.rank,
                device=lm_head.weight.device,
                dtype=lm_head.weight.dtype
            ) for _ in range(self.reweighting_params.num_iteration)
        ])
        self._setup_optimizer()
        custom_print("set_reweighting_heads", self.reweighting_heads)
        # custom_print("reweighting_head_optimizer", self.reweighting_head_optimizer)
    
    @torch.inference_mode(False)
    def _setup_optimizer(self):
        """Setup optimizer for independent weights"""
        # Optimize independent weights instead of original reweighting head parameters
        params_to_optimize = list(self.reweighting_heads.parameters())
        self.reweighting_head_optimizer = torch.optim.Adam(
            params_to_optimize,
            lr=self.reweighting_params.learning_rate
        )

    def _get_processor_idx_from_request_id(self, request_id: str) -> int:
        """Designate which processor to use for the given request_id"""
        # Parse request_id, e.g. "0_parent_id" -> 0
        if "_" in request_id:
            idx_str = request_id.split("_")[0]
            try:
                return int(idx_str)
            except ValueError:
                return 0
        return 0

    def forward(
        self,
        lm_head: VocabParallelEmbedding,
        hidden_states: torch.Tensor,
        sampling_metadata = None,
        embedding_bias: Optional[torch.Tensor] = None,
        request_ids: Optional[list] = None,
        num_tokens_per_request: Optional[list] = None,
    ) -> Optional[torch.Tensor]:
        if self.reweighting_heads is None:
            self.set_reweighting_heads(lm_head)
        
        if self.logits_as_input:
            logits = hidden_states
        else:
            logits = self._get_logits(hidden_states, lm_head, embedding_bias)
            if (self.reweighting_heads is not None and 
                request_ids is not None and 
                num_tokens_per_request is not None):
                logits, q_logits_next, q_indices = self._get_logits_with_multiple_heads_batched(hidden_states, logits, request_ids, num_tokens_per_request, embedding_bias)
            else:
                q_logits_next = None
                q_indices = None
            
            # logits = self._get_logits(hidden_states, lm_head, embedding_bias)
                
        if logits is not None:
            if self.soft_cap is not None:
                logits = logits / self.soft_cap
                logits = torch.tanh(logits)
                logits = logits * self.soft_cap

            if self.scale != 1.0:
                logits *= self.scale

        if q_logits_next is None:
            return logits
        else:
            return logits, q_logits_next, q_indices

    def _get_logits_with_multiple_heads_batched(
        self,
        hidden_states: torch.Tensor,
        logits: torch.Tensor,
        request_ids: list,
        num_tokens_per_request: list,
        embedding_bias: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        processor_idxs = [self._get_processor_idx_from_request_id(request_id) for request_id in request_ids]
        
        total_tokens = hidden_states.shape[0]
        global_indices = torch.arange(total_tokens, device=hidden_states.device)
        
        processor_groups = {}
        current_pos = 0
        for processor_idx, num_tokens in zip(processor_idxs, num_tokens_per_request):
            if processor_idx not in processor_groups:
                processor_groups[processor_idx] = []
            end_pos = current_pos + num_tokens
            processor_groups[processor_idx].append(global_indices[current_pos:end_pos])
            current_pos = end_pos
        
        all_logits = torch.zeros(total_tokens, self.vocab_size, 
                                device=hidden_states.device, dtype=hidden_states.dtype)
        # !! pre-allocate q_logits_next and indices would cause drastic efficiency degradation !!
        all_q_logits_next = []
        all_indices = []
        
        for processor_idx, indices_list in processor_groups.items():
            batch_indices = torch.cat(indices_list, dim=0)
            if batch_indices.shape[0] == 0:
                continue
            batch_hidden_states = hidden_states[batch_indices]
            batch_logits = logits[batch_indices]
            batch_reweighted_logits, batch_q_logits_next = self._compute_reweighted_logits(
                batch_hidden_states,
                embedding_bias,
                batch_logits,
                processor_idx + 1  # 1, 2, 3, ...
            )
            all_logits[batch_indices] = batch_reweighted_logits
            if batch_q_logits_next is not None:
                all_q_logits_next.append(batch_q_logits_next)
                all_indices.append(batch_indices)
        
        return all_logits, all_q_logits_next, all_indices

    def _get_reweighting_head(self, iteration: int):
        """Get the reweighting head for the given iteration"""
        if iteration < 1 or iteration >= self.reweighting_params.num_iteration:
            return None
        else:
            return self.reweighting_heads[iteration - 1]

    @torch.inference_mode(False)  # Disable inference mode to enable gradients
    def _apply_reweighting_head_with_grad(self, iteration, x, bias=None):
        with torch.enable_grad():
            # Clone inference tensors to convert them to normal tensors for autograd
            head = self._get_reweighting_head(iteration)
            x = x.detach().clone()  # Convert inference tensor to normal tensor
            out = head(x)
        return out

    @torch.inference_mode(False)  # Disable inference mode to enable gradients
    def _compute_reweighted_logits(self, hidden_states, embedding_bias, logits, current_iteration):
        # iteration: 1, 2, 3, ... --- idx: 0, 1, 2, ...
        last_iteration = current_iteration - 1
        # Reweighting mechanism
        with torch.no_grad():
            if self._get_reweighting_head(last_iteration) is None:
                p_logits = None
                reweighted_logits = logits
            else:
                p_logits = logits
                residual_logits = self._apply_reweighting_head_with_grad(last_iteration, hidden_states, embedding_bias)  # this is q_logits - p_logits
                reweighted_logits = p_logits - self.reweighting_params.lambd * residual_logits
        
        # Always compute q_logits_next, but track whether it should be used for updates
        has_reweighting_head = self._get_reweighting_head(current_iteration) is not None
        if has_reweighting_head:
            # Direct linear transformation using the weight matrix
            residual_logits_next = self._apply_reweighting_head_with_grad(current_iteration, hidden_states, embedding_bias)
            q_logits_next = logits.detach().clone() + residual_logits_next
        else:
            q_logits_next = None
        
        return reweighted_logits, q_logits_next

    @torch.inference_mode(False)  # Disable inference mode to enable gradients
    def update_reweighting_head(self, q_logits_next_list, sampled_token_ids, q_indices_list):
        """
        Update reweighting head with batched computation
        
        Args:
            q_logits_next_list: List of Q logits to compute loss on
            sampled_token_ids: The target token IDs
            q_indices_list: List of indices for each batch
        """
        if len(q_logits_next_list) == 0:
            return
        # Concatenate all q_logits_next from the list
        all_q_logits = torch.cat([q_logits.view(-1, q_logits.size(-1)) for q_logits in q_logits_next_list], dim=0)
        
        # Concatenate all target token ids
        all_target_ids = torch.cat([sampled_token_ids[q_indices].view(-1).long() for q_indices in q_indices_list], dim=0)
        
        # Compute loss for all batches at once
        reweighting_loss = F.cross_entropy(
            all_q_logits,
            all_target_ids,
            ignore_index=-100
        )
        
        self.reweighting_head_optimizer.zero_grad()
        reweighting_loss.backward()
        self.reweighting_head_optimizer.step()