

from dataclasses import dataclass
from typing import Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from vllm.model_executor.layers.logits_processor import LogitsProcessor, _prune_hidden_states, _apply_logits_processors

from contextlib import contextmanager
import time
import json
import copy
import random
from tqdm import tqdm
import numpy as np
import multiprocessing
import threading


@dataclass
class OverRIDEParams:
    """Parameters for OverRIDE reweighting mechanism"""
    lambd: float = 0.8
    num_iteration: int = 10
    rank: int = 1024
    batch_size: int = 256
    learning_rate: float = 1e-6


def custom_print(note, item=None):
    """Print a note and an item if provided"""
    print('='*100)
    print(note)
    if item is not None:
        print('-'*70)
        if isinstance(item, dict):
            print(json.dumps(item, indent=4))
        else:
            print(item)
    print('='*100)


def marked_timer(func):
    """Decorator to calculate running time of function and accumulate it"""
    def wrapper(self, *args, **kwargs):
        start_time = time.time()
        result = func(self, *args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        self.cumulative_training_time += elapsed_time
        return result
    return wrapper


class OverRIDELogitsProcessor(LogitsProcessor):
    def __init__(self,
                 vocab_size: int,
                 org_vocab_size: Optional[int] = None,
                 scale: float = 1.0,
                 logits_as_input: bool = False,
                 soft_cap: Optional[float] = None,
                 reweighting_params: OverRIDEParams=OverRIDEParams()) -> None:
        super().__init__(vocab_size, org_vocab_size, scale, logits_as_input, soft_cap)
        self.reweighting_params = reweighting_params
    
    def set_reweighting_params(self, reweighting_params: OverRIDEParams):
        self.reweighting_params = reweighting_params
    
    @torch.inference_mode(False)  # Disable inference mode to enable gradients
    def _init_reweighting_heads(self, lm_head, reweighting_params: OverRIDEParams=OverRIDEParams()):
        """Initialize reweighting heads as copies of lm_head"""
        if not hasattr(self, 'reweighting_params'):
            self.reweighting_params = reweighting_params
        # Try to get device index and add 3, else fallback to original device
        device = lm_head.weight.device
        target_device = f'cuda:{device.index + 3}' if device.type == 'cuda' and device.index is not None else device
        # Init reweighting heads
        self.reweighting_heads = nn.ParameterList([
            # nn.Parameter(lm_head.weight.data.clone(), requires_grad=True).to(lm_head.weight.device)
            nn.Parameter(lm_head.weight.data.clone(), requires_grad=True).to(target_device)
            for i in range(self.reweighting_params.num_iteration - 1)
        ])
        # Init training-related variables
        self.cumulative_training_time = 0.0
        self._setup_optimizer()
        self.loss_list = []
        self.loss_history = []
        self.current_iteration = 1
    
    @torch.inference_mode(False)
    def _setup_optimizer(self):
        """Setup optimizer for independent weights"""
        # Optimize independent weights instead of original reweighting head parameters
        params_to_optimize = list(self.reweighting_heads.parameters())
        self.reweighting_head_optimizer = torch.optim.Adam(
            params_to_optimize,
            lr=self.reweighting_params.learning_rate
        )
    
    def _check_and_update(self):
        """Check if loss list has reached batch_size and update if needed"""
        if len(self.loss_list) >= self.reweighting_params.batch_size:
            self.update_reweighting_head()
    
    @marked_timer
    @torch.inference_mode(False)
    def update_reweighting_head(self):
        """Explicitly average update loss list and update parameters with optimizer"""
        if len(self.loss_list) == 0:
            return
        avg_loss = torch.stack(self.loss_list).mean()
        self.reweighting_head_optimizer.zero_grad()
        avg_loss.backward()
        self.reweighting_head_optimizer.step()
        self.loss_history.append(avg_loss.item())
        self.loss_list = []  # Clear the loss list
        # print(f"Updated reweighting head with loss {avg_loss.item()}")
        return avg_loss.item()
    
    
    def set_reweighting_head(self, iteration: int):
        """Set the active reweighting head based on iteration"""
        self.current_iteration = iteration
    
    
    def _get_reweighting_head(self, iteration: int):
        """Get the reweighting head for the given iteration"""
        if iteration < 1 or iteration >= self.reweighting_params.num_iteration:
            return None
        else:
            return self.reweighting_heads[iteration - 1]
    
    def reset_training(self):
        """Print and return cumulative training time, then reset it to zero"""
        training_time = self.cumulative_training_time
        loss_history = copy.deepcopy(self.loss_history)
        print(f"Cumulative training time: {training_time:.4f} seconds")
        self.cumulative_training_time = 0.0
        self.loss_list = []
        self.loss_history = []
        return training_time, loss_history
    
    def _apply_reweighting_head_with_grad(self, iteration, x, bias=None):
        with torch.enable_grad():
            # Clone inference tensors to convert them to normal tensors for autograd
            weight = self._get_reweighting_head(iteration)
            x_normal = x.detach().clone().to(weight.device)  # Convert inference tensor to normal tensor
            out = torch.matmul(x_normal, weight.t())
            if bias is not None:
                out += bias
        return out.to(x.device)
    
    @marked_timer
    @torch.inference_mode(False)  # Disable inference mode to enable gradients
    def _compute_reweighted_logits(self, hidden_states, embedding_bias, logits, current_iteration):
        last_iteration = current_iteration - 1
        # Reweighting mechanism
        with torch.no_grad():
            if self._get_reweighting_head(last_iteration) is None:
                p_logits = None
                q_logits = None
                reweighted_logits = logits
            else:
                p_logits = logits
                # q_logits = self._get_logits(hidden_states, last_reweighting_head, embedding_bias)
                q_logits = self._apply_reweighting_head_with_grad(last_iteration, hidden_states, embedding_bias)
                reweighted_logits = self.reweighting_params.lambd * (p_logits - q_logits) + p_logits
                # reweighted_logits = p_logits - q_logits
        
        if self._get_reweighting_head(current_iteration) is not None:
            # Direct linear transformation using the weight matrix
            q_logits_next = self._apply_reweighting_head_with_grad(current_iteration, hidden_states, embedding_bias)
            # custom_print(
            #     "logits info",
            #     {
            #         "logits": str([logits.requires_grad, logits[0, :10].tolist()]),
            #         "q_logits": None if q_logits is None else str([q_logits.requires_grad, q_logits[0, :10].tolist()]),
            #         "q_logits_next": str([q_logits_next.requires_grad, q_logits_next[0, :10].tolist()]),
            #         "diff": (logits - q_logits_next).norm(p=2).item() if q_logits is None else (logits - q_logits).norm(p=2).item(),
            #         "max_diff": (logits - q_logits_next).abs().max().item() if q_logits is None else (logits - q_logits).abs().max().item(),
            #         "current_iteration": current_iteration,
            #         "last_iteration": last_iteration,
            #         "logits_info": {"min": logits.min().item(), "max": logits.max().item(), "mean": logits.mean().item(), "std": logits.std().item()},
            #         "q_logits_info": None if q_logits is None else {"min": q_logits.min().item(), "max": q_logits.max().item(), "mean": q_logits.mean().item(), "std": q_logits.std().item()},
            #         "q_logits_next_info": {"min": q_logits_next.min().item(), "max": q_logits_next.max().item(), "mean": q_logits_next.mean().item(), "std": q_logits_next.std().item()},
            #     }
            # )

            # Compute cross-entropy loss
            predicted_tokens = torch.argmax(reweighted_logits, dim=-1).detach()
            reweighting_loss = F.cross_entropy(
                q_logits_next.view(-1, q_logits_next.size(-1)),
                predicted_tokens.view(-1),
                ignore_index=-100
            )
            
            # # Compute KL divergence loss using log_softmax
            # reweighted_log_probs = F.log_softmax(reweighted_logits, dim=-1).detach()
            # q_logits_next_log_probs = F.log_softmax(q_logits_next, dim=-1)
            # reweighting_loss = F.kl_div(
            #     q_logits_next_log_probs,    # Input should be log-probabilities
            #     reweighted_log_probs,       # Target should be log-probabilities
            #     reduction='batchmean',      # Average over batch dimension
            #     log_target=True             # Enable log_target for numerical stability
            # )

            # Update loss list
            self.loss_list.append(reweighting_loss)
            self._check_and_update()

            # self.reweighting_head_optimizer.zero_grad()
            # reweighting_loss.backward()
            # self.reweighting_head_optimizer.step()
            # self.loss_history.append(reweighting_loss.item())
            # # custom_print(f"Updated reweighting head with reweighting_loss", reweighting_loss.item())
        
        return reweighted_logits

    def forward(
        self,
        lm_head,
        hidden_states,
        sampling_metadata = None,
        embedding_bias = None,
        iteration = None,
    ):
        # Init reweighting heads with the original lm_head
        if not hasattr(self, 'reweighting_heads'):
            # custom_print("lm_head params", [(name, param.shape) for name, param in lm_head.named_parameters()])  # [('weight', torch.Size([128256, 3072]))]
            self._init_reweighting_heads(lm_head)
        
        # Get current reweighting head
        if iteration is None:
            current_iteration = self.current_iteration
        
        if self.logits_as_input:
            logits = hidden_states
            raise NotImplementedError("logits_as_input is not supported for OverRIDE")
        else:
            if sampling_metadata is not None:
                hidden_states = _prune_hidden_states(hidden_states,
                                                     sampling_metadata)

            # Get the logits for the next tokens.
            logits = self._get_logits(hidden_states, lm_head, embedding_bias)
        
        # OverRIDE reweighting
        logits = self._compute_reweighted_logits(
            hidden_states,
            embedding_bias,
            logits,
            current_iteration
        )

        if logits is not None:
            if self.soft_cap is not None:
                logits = logits / self.soft_cap
                logits = torch.tanh(logits)
                logits = logits * self.soft_cap

            if self.scale != 1.0:
                logits *= self.scale

            # Apply logits processors (if any).
            if sampling_metadata is not None and \
                sampling_metadata.seq_groups is not None:
                logits = _apply_logits_processors(logits, sampling_metadata)

        return logits