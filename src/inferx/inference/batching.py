"""
Batch preparation, tokenization, and alignment utilities for InferX.
"""

from typing import Dict, List, Union
import torch
from transformers import PreTrainedTokenizer

from inferx.utils.logging import logger


class BatchManager:
    """Manages input batching, left-padding, truncation, and tensor alignment."""

    def __init__(self, tokenizer: PreTrainedTokenizer, device: str):
        self.tokenizer = tokenizer
        self.device = device
        # Causal language models require left padding for autoregressive generation
        self.tokenizer.padding_side = "left"

    def prepare_batch(
        self,
        prompts: Union[str, List[str]],
        max_length: int = 512,
    ) -> Dict[str, torch.Tensor]:
        """Tokenize and pad a collection of prompts into a uniform batch tensor.

        Args:
            prompts: Single string or list of input strings.
            max_length: Maximum allowed sequence token length.

        Returns:
            Dictionary containing 'input_ids' and 'attention_mask' on target device.
        """
        if isinstance(prompts, str):
            prompts = [prompts]

        encoded = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        # Move to target device
        batch = {
            "input_ids": encoded["input_ids"].to(self.device),
            "attention_mask": encoded["attention_mask"].to(self.device),
        }
        return batch

    def calculate_batch_throughput(
        self,
        batch_size: int,
        generated_tokens_per_sample: int,
        latency_seconds: float,
    ) -> Dict[str, float]:
        """Compute serving throughput metrics for a completed batch run.

        Args:
            batch_size: Number of concurrent sequences processed.
            generated_tokens_per_sample: Number of new tokens generated per sequence.
            latency_seconds: Wall-clock duration in seconds.

        Returns:
            Dictionary containing 'tokens_per_second' and 'requests_per_second'.
        """
        if latency_seconds <= 0:
            return {"tokens_per_second": 0.0, "requests_per_second": 0.0}

        total_tokens = batch_size * generated_tokens_per_sample
        tokens_per_second = total_tokens / latency_seconds
        requests_per_second = batch_size / latency_seconds

        return {
            "tokens_per_second": round(tokens_per_second, 2),
            "requests_per_second": round(requests_per_second, 2),
        }
