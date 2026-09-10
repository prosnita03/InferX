"""
Statistical latency benchmarking engine for InferX.
"""

from dataclasses import asdict, dataclass
import time
from typing import Any, Dict, List, Optional
import numpy as np

from inferx.inference.engine import InferenceEngine
from inferx.utils.logging import logger
from inferx.utils.system import format_latency, sync_device


@dataclass
class LatencyMetrics:
    """Rigorous statistical metrics computed across benchmark iterations."""
    warmup_iterations: int
    benchmark_iterations: int
    model_load_time_ms: float
    warmup_time_total_ms: float
    mean_latency_ms: float
    median_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    std_latency_ms: float
    p50_latency_ms: float
    p90_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    time_to_first_token_ms: Optional[float]
    latencies_raw_ms: List[float]

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return asdict(self)

    def summary(self) -> str:
        """Format an executive statistical latency summary."""
        ttft_str = f" | TTFT: {format_latency(self.time_to_first_token_ms)}" if self.time_to_first_token_ms else ""
        return (
            f"Latency Statistics ({self.benchmark_iterations} iterations):\n"
            f"  Mean:   {format_latency(self.mean_latency_ms)} (± {format_latency(self.std_latency_ms)})\n"
            f"  Median: {format_latency(self.median_latency_ms)} | Min: {format_latency(self.min_latency_ms)} | Max: {format_latency(self.max_latency_ms)}\n"
            f"  P50:    {format_latency(self.p50_latency_ms)} | P90: {format_latency(self.p90_latency_ms)} | P95: {format_latency(self.p95_latency_ms)} | P99: {format_latency(self.p99_latency_ms)}{ttft_str}\n"
            f"  Model Load Time: {format_latency(self.model_load_time_ms)} | Warmup Total: {format_latency(self.warmup_time_total_ms)}"
        )


class LatencyBenchmark:
    """Executes controlled latency benchmarking with CUDA/device synchronization."""

    def __init__(self, engine: InferenceEngine):
        """Initialize latency benchmark runner with an active InferenceEngine.

        Args:
            engine: Configured InferenceEngine instance.
        """
        self.engine = engine
        self.device = engine.device

    def run(
        self,
        prompts: List[str],
        max_new_tokens: int = 64,
        warmup_iterations: int = 3,
        benchmark_iterations: int = 10,
        use_cache: bool = True,
        model_load_time_ms: float = 0.0,
        measure_ttft: bool = True,
    ) -> LatencyMetrics:
        """Execute warmup passes, steady-state benchmark iterations, and compute percentiles.

        Args:
            prompts: Input text prompt or batch of prompts.
            max_new_tokens: Token count to generate.
            warmup_iterations: Iteration count to discard before measuring.
            benchmark_iterations: Iteration count for statistical evaluation.
            use_cache: KV cache enabled flag.
            model_load_time_ms: Wall-clock time taken during model load.
            measure_ttft: Whether to profile prefill TTFT on a single sample.

        Returns:
            LatencyMetrics dataclass.
        """
        logger.info(
            f"Starting Latency Benchmark: {warmup_iterations} warmups, "
            f"{benchmark_iterations} iterations, max_new_tokens={max_new_tokens}"
        )

        is_batch = len(prompts) > 1

        # 1. Warm-up phase
        warmup_start = time.perf_counter()
        for i in range(warmup_iterations):
            if is_batch:
                _ = self.engine.generate_batch(
                    prompts=prompts,
                    max_new_tokens=min(max_new_tokens, 16),
                    do_sample=False,
                    use_cache=use_cache,
                )
            else:
                _ = self.engine.generate(
                    prompt=prompts[0],
                    max_new_tokens=min(max_new_tokens, 16),
                    do_sample=False,
                    use_cache=use_cache,
                )
        sync_device(self.device)
        warmup_total_ms = (time.perf_counter() - warmup_start) * 1000.0

        # 2. Steady-state benchmark phase
        latencies_ms: List[float] = []

        for i in range(benchmark_iterations):
            sync_device(self.device)
            iter_start = time.perf_counter()

            if is_batch:
                _ = self.engine.generate_batch(
                    prompts=prompts,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=use_cache,
                )
            else:
                _ = self.engine.generate(
                    prompt=prompts[0],
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=use_cache,
                )

            sync_device(self.device)
            iter_end = time.perf_counter()
            latencies_ms.append((iter_end - iter_start) * 1000.0)

        # 3. TTFT measurement (optional detailed profile on single prompt)
        ttft_ms = None
        if measure_ttft and not is_batch:
            try:
                single_res = self.engine.measure_token_latencies(
                    prompt=prompts[0],
                    max_new_tokens=min(max_new_tokens, 16),
                    do_sample=False,
                    use_cache=use_cache,
                )
                ttft_ms = single_res.time_to_first_token_ms
            except Exception as e:
                logger.debug(f"TTFT step measurement skipped: {e}")

        # 4. Statistical Percentile Computations
        arr = np.array(latencies_ms)
        mean_ms = float(np.mean(arr))
        median_ms = float(np.median(arr))
        min_ms = float(np.min(arr))
        max_ms = float(np.max(arr))
        std_ms = float(np.std(arr))
        p50_ms = float(np.percentile(arr, 50))
        p90_ms = float(np.percentile(arr, 90))
        p95_ms = float(np.percentile(arr, 95))
        p99_ms = float(np.percentile(arr, 99))

        metrics = LatencyMetrics(
            warmup_iterations=warmup_iterations,
            benchmark_iterations=benchmark_iterations,
            model_load_time_ms=round(model_load_time_ms, 2),
            warmup_time_total_ms=round(warmup_total_ms, 2),
            mean_latency_ms=round(mean_ms, 2),
            median_latency_ms=round(median_ms, 2),
            min_latency_ms=round(min_ms, 2),
            max_latency_ms=round(max_ms, 2),
            std_latency_ms=round(std_ms, 2),
            p50_latency_ms=round(p50_ms, 2),
            p90_latency_ms=round(p90_ms, 2),
            p95_latency_ms=round(p95_ms, 2),
            p99_latency_ms=round(p99_ms, 2),
            time_to_first_token_ms=round(ttft_ms, 2) if ttft_ms is not None else None,
            latencies_raw_ms=[round(x, 2) for x in latencies_ms],
        )

        logger.info(f"Latency benchmark finished:\n{metrics.summary()}")
        return metrics
