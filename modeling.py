from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from contextlib import contextmanager
from codetiming import Timer
import time
import json

# LogitsProcessor
from vllm.model_executor.layers.logits_processor import LogitsProcessor, _prune_hidden_states, _apply_logits_processors
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.sampling_metadata import SamplingMetadata

@dataclass
class OverRIDEParams:
    """Parameters for OverRIDE reweighting mechanism"""
    lambd: float = 50.0
    num_iteration: int = 10
    rank: int = 16
    batch_size: int = 256
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


def _timer(name: str, timing_raw: dict[str, float]):
    """Inner function that handles the core timing logic.

    Args:
        name (str): The name/identifier for this timing measurement.
        timing_raw (Dict[str, float]): Dictionary to store timing information.
    """
    with Timer(name=name, logger=None) as timer:
        yield
    if name not in timing_raw:
        timing_raw[name] = 0
    timing_raw[name] += timer.last


@contextmanager
def marked_timer(
    name: str,
    timing_raw: dict[str, float],
    color: str = None,
    domain: Optional[str] = None,
    category: Optional[str] = None,
):
    """Context manager for timing with platform markers.

    This utility function measures the execution time of code within its context,
    accumulates the timing information, and adds platform markers for profiling.
    This function is a default implementation when hardware profiler is not available.

    Args:
        name (str): The name/identifier for this timing measurement.
        timing_raw (Dict[str, float]): Dictionary to store timing information.
        color (Optional[str]): Color for the marker. Defaults to None.
        domain (Optional[str]): Domain for the marker. Defaults to None.
        category (Optional[str]): Category for the marker. Defaults to None.

    Yields:
        None: This is a context manager that yields control back to the code block.
    """
    yield from _timer(name, timing_raw)


def simple_timer(func):
    """Decorator to calculate running time of function and accumulate it"""
    def wrapper(self, *args, **kwargs):
        start_time = time.time()
        result = func(self, *args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        self.cumulative_training_time += elapsed_time
        return result
    return wrapper


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
    """Process logits and apply logits processors from sampling metadata.

    This layer does the following:
    1. Gather logits from model hidden_states.
    2. Scale logits if needed.
    3. Apply logits processors (if any).
    """

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
        """根据request_id确定使用哪个processor"""
        # 解析request_id，例如 "0_parent_id" -> 0
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
        sampling_metadata: Optional[SamplingMetadata] = None,
        embedding_bias: Optional[torch.Tensor] = None,
        request_ids: Optional[list] = None,
        num_tokens_per_request: Optional[list] = None,
    ) -> Optional[torch.Tensor]:
        if self.reweighting_heads is None:
            self.set_reweighting_heads(lm_head)
        
        if self.logits_as_input:
            logits = hidden_states
        else:
            if sampling_metadata is not None:
                hidden_states = _prune_hidden_states(hidden_states,
                                                     sampling_metadata)

            # 检查是否使用多个lm_head
            logits = self._get_logits(hidden_states, lm_head, embedding_bias)
            if (self.reweighting_heads is not None and 
                request_ids is not None and 
                num_tokens_per_request is not None):
                logits = self._get_logits_with_multiple_heads_batched(hidden_states, logits, request_ids, num_tokens_per_request, embedding_bias)
            
            # logits = self._get_logits(hidden_states, lm_head, embedding_bias)
                
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

    def _get_logits_with_multiple_heads(
        self,
        hidden_states: torch.Tensor,
        request_ids: list,
        num_tokens_per_request: list,
        embedding_bias: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """使用多个lm_head计算logits，正确处理不同request的token数量"""
        logits_list = []
        current_pos = 0
        
        processor_idxs = [self._get_processor_idx_from_request_id(request_id) for request_id in request_ids]
        # custom_print("processor_idx", processor_idxs)
        for processor_idx, num_tokens in zip(processor_idxs, num_tokens_per_request):
            lm_head = self.reweighting_heads[processor_idx]
            req_hidden_states = hidden_states[current_pos:current_pos + num_tokens]
            req_logits = self._get_logits(req_hidden_states, lm_head, embedding_bias)
            logits_list.append(req_logits)
            current_pos += num_tokens
        
        return torch.cat(logits_list, dim=0)

    def _get_logits_with_multiple_heads_batched(
        self,
        hidden_states: torch.Tensor,
        logits: torch.Tensor,
        request_ids: list,
        num_tokens_per_request: list,
        embedding_bias: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """使用多个lm_head批量计算logits，通过索引操作优化性能"""
        processor_idxs = [self._get_processor_idx_from_request_id(request_id) for request_id in request_ids]
        
        # 创建全局索引映射
        total_tokens = hidden_states.shape[0]
        global_indices = torch.arange(total_tokens, device=hidden_states.device)
        
        # 按processor_idx分组索引
        processor_groups = {}
        current_pos = 0
        for processor_idx, num_tokens in zip(processor_idxs, num_tokens_per_request):
            if processor_idx not in processor_groups:
                processor_groups[processor_idx] = []
            end_pos = current_pos + num_tokens
            processor_groups[processor_idx].append(global_indices[current_pos:end_pos])
            current_pos = end_pos
        
        # 预分配结果张量
        all_logits = torch.zeros(total_tokens, self.vocab_size, 
                                device=hidden_states.device, dtype=hidden_states.dtype)
        
        # 批量处理每个processor_idx
        for processor_idx, indices_list in processor_groups.items():
            batch_indices = torch.cat(indices_list, dim=0)
            if batch_indices.shape[0] == 0:
                continue
            batch_hidden_states = hidden_states[batch_indices]
            batch_logits = logits[batch_indices]
            all_logits[batch_indices] = self._compute_reweighted_logits(
                batch_hidden_states,
                embedding_bias,
                batch_logits,
                processor_idx + 1  # 1, 2, 3, ...
            )
        
        return all_logits

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
        
        if self._get_reweighting_head(current_iteration) is not None:
            # Direct linear transformation using the weight matrix
            residual_logits_next = self._apply_reweighting_head_with_grad(current_iteration, hidden_states, embedding_bias)
            q_logits_next = logits.detach().clone() + residual_logits_next

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

            # Update reweighting head
            self.reweighting_head_optimizer.zero_grad()
            reweighting_loss.backward()
            self.reweighting_head_optimizer.step()
            # self.loss_history.append(reweighting_loss.item())
            # custom_print(f"Updated reweighting head with reweighting_loss", reweighting_loss.item())

        return reweighted_logits