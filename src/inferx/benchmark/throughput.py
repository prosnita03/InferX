"""
High-concurrency throughput benchmarking suite for InferX.
"""

from dataclasses import asdict, dataclass
import time
from typing import Any, Dict, List, Optional
import torch

from inferx.inference.engine import InferenceEngine
from inferx.utils.logging import logger
from inferx.utils.system import format_throughput, sync_device


@dataclass
class ThroughputResult:
    """Throughput measurement for a specific batch size configuration."""
    batch_size: int
    generated_tokens_total: int
    prompt_tokens_total: int
    latency_seconds: float
    tokens_per_second: float
    requests_per_second: float
    total_tokens_per_second: float
    status: str  # "SUCCESS", "OOM", "FAILED"
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert throughput record to dictionary."""
        return asdict(self)

    def summary(self) -> str:
        """Format an executive throughput summary."""
        if self.status != "SUCCESS":
            return f"Batch Size {self.batch_size:2d}: FAILED ({self.status}: {self.error_message})"
        return (
            f"Batch Size {self.batch_size:2d}: {format_throughput(self.tokens_per_second)} | "
            f"{self.requests_per_second:.2f} req/s | Total Latency: {self.latency_seconds * 1000:.1f} ms"
        )


class ThroughputBenchmark:
    """Evaluates generation throughput across varying concurrency and batch scales."""

    def __init__(self, engine: InferenceEngine):
        """Initialize throughput benchmark runner.

        Args:
            engine: Configured InferenceEngine instance.
        """
        self.engine = engine
        self.device = engine.device

    def run_single_batch(
        self,
        batch_size: int,
        prompt: str = "Explain the fundamental difference between latency and throughput in distributed machine learning systems.",
        max_new_tokens: int = 64,
        use_cache: bool = True,
    ) -> ThroughputResult:
        """Execute a single batch throughput measurement with graceful OOM handling.

        Args:
            batch_size: Number of parallel sequences.
            prompt: Text prompt replicated across the batch.
            max_new_tokens: Generation budget per sequence.
            use_cache: KV cache enabled flag.

        Returns:
            ThroughputResult instance.
        """
        prompts = [prompt] * batch_size

        try:
            sync_device(self.device)
            start_time = time.perf_counter()

            results = self.engine.generate_batch(
                prompts=prompts,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=use_cache,
            )

            sync_device(self.device)
            end_time = time.perf_counter()
            duration_s = end_time - start_time

            total_generated_tokens = sum(r.generated_tokens for r in results)
            total_prompt_tokens = sum(r.prompt_tokens for r in results)

            tokens_per_sec = total_generated_tokens / duration_s if duration_s > 0 else 0.0
            requests_per_sec = batch_size / duration_s if duration_s > 0 else 0.0
            total_tokens_per_sec = (total_generated_tokens + total_prompt_tokens) / duration_s if duration_s > 0 else 0.0

            res = ThroughputResult(
                batch_size=batch_size,
                generated_tokens_total=total_generated_tokens,
                prompt_tokens_total=total_prompt_tokens,
                latency_seconds=round(duration_s, 4),
                tokens_per_second=round(tokens_per_sec, 2),
                requests_per_second=round(requests_per_sec, 2),
                total_tokens_per_second=round(total_tokens_per_sec, 2),
                status="SUCCESS",
            )
            logger.info(res.summary())
            return res

        except (torch.cuda.OutOfMemoryError, MemoryError) as e:
            logger.warning(f"Batch size {batch_size} exceeded memory limits (OOM): {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return ThroughputResult(
                batch_size=batch_size,
                generated_tokens_total=0,
                prompt_tokens_total=0,
                latency_seconds=0.0,
                tokens_per_second=0.0,
                requests_per_second=0.0,
                total_tokens_per_second=0.0,
                status="OOM",
                error_message=str(e),
            )
        except Exception as e:
            logger.error(f"Batch size {batch_size} encountered error: {e}")
            return ThroughputResult(
                batch_size=batch_size,
                generated_tokens_total=0,
                prompt_tokens_total=0,
                latency_seconds=0.0,
                tokens_per_second=0.0,
                requests_per_second=0.0,
                total_tokens_per_second=0.0,
                status="FAILED",
                error_message=str(e),
            )

    def run_sweep(
        self,
        batch_sizes: List[int] = [1, 2, 4, 8],
        prompt: str = "Explain the fundamental difference between latency and throughput in distributed machine learning systems.",
        max_new_tokens: int = 64,
        use_cache: bool = True,
    ) -> List[ThroughputResult]:
        """Sweep across multiple batch sizes to map the throughput scaling curve.

        Args:
            batch_sizes: List of batch sizes to evaluate.
            prompt: Text prompt string.
            max_new_tokens: Generated tokens count.
            use_cache: KV cache enabled flag.

        Returns:
            List of ThroughputResult objects.
        """
        logger.info(f"Running throughput sweep across batch sizes: {batch_sizes}")
        results = []
        for bs in batch_sizes:
            res = self.run_single_batch(
                batch_size=bs,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                use_cache=use_cache,
            )
            results.append(res)
            # If OOM occurs, avoid running even larger batch sizes in the sweep
            if res.status == "OOM":
                logger.warning(f"Halting throughput sweep at batch size {bs} due to OOM.")
                break
        return results
