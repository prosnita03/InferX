"""
Core high-performance inference engine for InferX.
"""

import time
from typing import Dict, List, Optional, Union
import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizer

from inferx.inference.batching import BatchManager
from inferx.inference.generation import GenerationResult
from inferx.utils.logging import logger
from inferx.utils.system import sync_device


class InferenceEngine:
    """Production inference execution engine supporting batched and fine-grained token profiling."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: Optional[str] = None,
    ):
        """Initialize the inference engine with loaded model and tokenizer.

        Args:
            model: PyTorch pre-trained language model.
            tokenizer: PreTrainedTokenizer instance.
            device: Execution device ('cuda', 'mps', 'cpu'). If None, detected from model parameters.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or str(next(model.parameters()).device)
        self.batch_manager = BatchManager(tokenizer=self.tokenizer, device=self.device)
        self.model.eval()

    def warmup(self, iterations: int = 2, max_new_tokens: int = 8) -> None:
        """Run warm-up passes to populate hardware caches and compile execution kernels.

        Args:
            iterations: Number of warmup iterations.
            max_new_tokens: Small token budget for warm-up.
        """
        logger.debug(f"Executing {iterations} warm-up iteration(s)...")
        prompt = "System initialization warmup benchmark."
        for _ in range(iterations):
            with torch.no_grad():
                inputs = self.batch_manager.prepare_batch([prompt], max_length=64)
                sync_device(self.device)
                _ = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                sync_device(self.device)
        logger.debug("Warm-up complete.")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = False,
        use_cache: bool = True,
    ) -> GenerationResult:
        """Run single-prompt inference with high-resolution latency and token tracking.

        Args:
            prompt: Text prompt string.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (ignored if do_sample is False).
            top_p: Nucleus sampling probability.
            top_k: Top-k vocabulary filtering.
            do_sample: Whether to sample or execute greedy deterministic decoding.
            use_cache: Whether to use past Key/Value state caching.

        Returns:
            GenerationResult dataclass with timing metrics and generated text.
        """
        inputs = self.batch_manager.prepare_batch([prompt], max_length=2048)
        input_ids = inputs["input_ids"]
        prompt_len = input_ids.shape[1]

        generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "use_cache": use_cache,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if do_sample:
            generate_kwargs["temperature"] = temperature
            generate_kwargs["top_p"] = top_p
            if top_k > 0:
                generate_kwargs["top_k"] = top_k

        # Synchronize before starting timer to isolate GPU kernel queue
        sync_device(self.device)
        start_time = time.perf_counter()

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                **generate_kwargs,
            )

        sync_device(self.device)
        end_time = time.perf_counter()

        total_latency_ms = (end_time - start_time) * 1000.0
        generated_token_ids = output_ids[0, prompt_len:]
        num_generated = len(generated_token_ids)
        decoded_text = self.tokenizer.decode(generated_token_ids, skip_special_tokens=True)

        return GenerationResult(
            prompt=prompt,
            generated_text=decoded_text,
            prompt_tokens=prompt_len,
            generated_tokens=num_generated,
            total_tokens=prompt_len + num_generated,
            total_latency_ms=total_latency_ms,
            time_to_first_token_ms=0.0,
            decode_latency_ms=total_latency_ms,
            inter_token_latencies_ms=[],
        )

    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 64,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = False,
        use_cache: bool = True,
    ) -> List[GenerationResult]:
        """Execute batched inference with left-padding and collective timing.

        Args:
            prompts: List of prompt strings.
            max_new_tokens: Number of tokens to generate per sequence.
            temperature: Sampling temperature.
            top_p: Top-p sampling parameter.
            top_k: Top-k parameter.
            do_sample: Sampling vs greedy decoding.
            use_cache: KV cache enabled flag.

        Returns:
            List of GenerationResult objects corresponding to each prompt in the batch.
        """
        inputs = self.batch_manager.prepare_batch(prompts, max_length=2048)
        input_ids = inputs["input_ids"]
        prompt_lens = [len(self.tokenizer.encode(p, add_special_tokens=False)) for p in prompts]

        generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "use_cache": use_cache,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if do_sample:
            generate_kwargs["temperature"] = temperature
            generate_kwargs["top_p"] = top_p
            if top_k > 0:
                generate_kwargs["top_k"] = top_k

        sync_device(self.device)
        start_time = time.perf_counter()

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                **generate_kwargs,
            )

        sync_device(self.device)
        end_time = time.perf_counter()

        total_latency_ms = (end_time - start_time) * 1000.0
        results = []

        for idx, (p, p_len) in enumerate(zip(prompts, prompt_lens)):
            # With left padding, generated tokens start after inputs["input_ids"].shape[1]
            generated_token_ids = output_ids[idx, inputs["input_ids"].shape[1]:]
            num_generated = len(generated_token_ids)
            decoded_text = self.tokenizer.decode(generated_token_ids, skip_special_tokens=True)

            results.append(
                GenerationResult(
                    prompt=p,
                    generated_text=decoded_text,
                    prompt_tokens=p_len,
                    generated_tokens=num_generated,
                    total_tokens=p_len + num_generated,
                    total_latency_ms=total_latency_ms,
                    time_to_first_token_ms=0.0,
                    decode_latency_ms=total_latency_ms,
                    inter_token_latencies_ms=[],
                )
            )

        return results

    def measure_token_latencies(
        self,
        prompt: str,
        max_new_tokens: int = 32,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = False,
        use_cache: bool = True,
    ) -> GenerationResult:
        """Autoregressively decode step-by-step to accurately measure TTFT and Inter-Token Latencies (ITL).

        Provides ground-truth breakdown:
        - Prefill phase (Time-To-First-Token)
        - Per-token autoregressive generation latencies
        """
        inputs = self.batch_manager.prepare_batch([prompt], max_length=2048)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        prompt_len = input_ids.shape[1]

        generated_ids: List[int] = []
        inter_token_latencies: List[float] = []
        past_key_values = None

        sync_device(self.device)
        overall_start = time.perf_counter()

        # Step 1: Prefill phase (computes First Token)
        prefill_start = time.perf_counter()
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=use_cache,
            )
            next_token_logits = outputs.logits[:, -1, :]

            if do_sample and temperature > 0:
                probs = torch.softmax(next_token_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            next_token_id = next_token.item()
            generated_ids.append(next_token_id)
            if use_cache:
                past_key_values = outputs.past_key_values

        sync_device(self.device)
        prefill_end = time.perf_counter()
        ttft_ms = (prefill_end - prefill_start) * 1000.0

        current_input_ids = next_token if use_cache else torch.cat([input_ids, next_token], dim=1)
        current_attention_mask = torch.cat(
            [attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=self.device)],
            dim=1,
        )

        # Step 2: Autoregressive decoding phase for remaining tokens
        decode_start = time.perf_counter()
        for _ in range(1, max_new_tokens):
            if next_token_id == self.tokenizer.eos_token_id:
                break

            step_start = time.perf_counter()
            with torch.no_grad():
                if use_cache:
                    outputs = self.model(
                        input_ids=current_input_ids,
                        attention_mask=current_attention_mask,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                    past_key_values = outputs.past_key_values
                else:
                    outputs = self.model(
                        input_ids=current_input_ids,
                        attention_mask=current_attention_mask,
                        use_cache=False,
                    )

                next_token_logits = outputs.logits[:, -1, :]
                if do_sample and temperature > 0:
                    probs = torch.softmax(next_token_logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                next_token_id = next_token.item()
                generated_ids.append(next_token_id)

                if use_cache:
                    current_input_ids = next_token
                else:
                    current_input_ids = torch.cat([current_input_ids, next_token], dim=1)

                current_attention_mask = torch.cat(
                    [current_attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=self.device)],
                    dim=1,
                )

            sync_device(self.device)
            step_end = time.perf_counter()
            inter_token_latencies.append((step_end - step_start) * 1000.0)

        overall_end = time.perf_counter()
        total_latency_ms = (overall_end - overall_start) * 1000.0
        decode_latency_ms = (overall_end - prefill_end) * 1000.0

        decoded_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return GenerationResult(
            prompt=prompt,
            generated_text=decoded_text,
            prompt_tokens=prompt_len,
            generated_tokens=len(generated_ids),
            total_tokens=prompt_len + len(generated_ids),
            total_latency_ms=total_latency_ms,
            time_to_first_token_ms=ttft_ms,
            decode_latency_ms=decode_latency_ms,
            inter_token_latencies_ms=inter_token_latencies,
        )
