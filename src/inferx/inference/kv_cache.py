"""
KV Cache management, profiling, and comparative analysis.

In autoregressive transformer decoding:
- Without KV cache (`use_cache=False`): At each generation step t, the model must
  recompute Key and Value projections for all tokens from 0 to t-1. Computational
  complexity per generated token scales quadratically with sequence length O(N^2).
- With KV cache (`use_cache=True`): Key and Value projections for previous tokens
  are retained in GPU/RAM memory tensors. Only the single newly sampled token is
  projected into Q, K, V and appended to the cache, achieving linear complexity O(N).
  Trade-off: Increased memory consumption for holding cache tensors.
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class KVCacheComparison:
    """Quantitative comparison of inference with KV-cache enabled vs disabled."""
    model_name: str
    tokens_generated: int
    cache_enabled_latency_ms: float
    cache_disabled_latency_ms: float
    cache_enabled_throughput: float
    cache_disabled_throughput: float
    cache_enabled_memory_mb: float
    cache_disabled_memory_mb: float

    @property
    def speedup_ratio(self) -> float:
        """Calculate generation speedup ratio achieved by KV caching."""
        if self.cache_enabled_latency_ms <= 0:
            return 1.0
        return round(self.cache_disabled_latency_ms / self.cache_enabled_latency_ms, 2)

    @property
    def memory_overhead_mb(self) -> float:
        """Memory cost incurred by holding past key-value states in cache."""
        return round(self.cache_enabled_memory_mb - self.cache_disabled_memory_mb, 2)

    def summary(self) -> str:
        """Human-readable explanation of the KV-cache trade-off."""
        return (
            f"KV Cache Experiment Results ({self.model_name}):\n"
            f"  Enabled Latency:  {self.cache_enabled_latency_ms:.2f} ms ({self.cache_enabled_throughput:.1f} tok/s)\n"
            f"  Disabled Latency: {self.cache_disabled_latency_ms:.2f} ms ({self.cache_disabled_throughput:.1f} tok/s)\n"
            f"  Speedup:          {self.speedup_ratio}x faster with KV cache\n"
            f"  Memory Overhead:  {self.memory_overhead_mb:+.2f} MB"
        )
