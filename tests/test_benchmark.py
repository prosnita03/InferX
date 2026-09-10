"""
Unit tests for InferX Benchmarking Suite (Hardware, Memory, Latency, Throughput).
"""

from unittest.mock import MagicMock, patch
import pytest
import torch

from inferx.benchmark.hardware import HardwareProfiler
from inferx.benchmark.latency import LatencyBenchmark, LatencyMetrics
from inferx.benchmark.memory import MemoryTracker
from inferx.benchmark.throughput import ThroughputBenchmark
from inferx.inference.batching import BatchManager


def test_hardware_profiler():
    profile = HardwareProfiler.profile()

    assert profile.os in ["Darwin", "Linux", "Windows"]
    assert profile.python_version != ""
    assert profile.pytorch_version != ""
    assert profile.system_ram_gb > 0
    assert profile.cpu_cores_logical >= 1

    summary = profile.summary()
    assert "Hardware Environment:" in summary
    assert "PyTorch:" in summary


def test_memory_tracker():
    with MemoryTracker(device="cpu") as tracker:
        # Allocate temporary tensor
        x = torch.ones((1000, 1000), dtype=torch.float32)
        _ = x.sum()

    snapshot = tracker.get_snapshot()
    assert snapshot.baseline_mb >= 0
    assert snapshot.peak_mb >= snapshot.baseline_mb
    assert snapshot.device == "cpu"
    assert "Baseline:" in snapshot.summary()


def test_batch_throughput_calculation():
    tokenizer = MagicMock()
    bm = BatchManager(tokenizer=tokenizer, device="cpu")

    res = bm.calculate_batch_throughput(
        batch_size=4,
        generated_tokens_per_sample=32,
        latency_seconds=2.0,
    )

    # 4 * 32 = 128 tokens / 2.0s = 64.0 tok/s
    assert res["tokens_per_second"] == 64.0
    # 4 requests / 2.0s = 2.0 req/s
    assert res["requests_per_second"] == 2.0


def test_latency_metrics_dataclass():
    metrics = LatencyMetrics(
        warmup_iterations=2,
        benchmark_iterations=5,
        model_load_time_ms=150.0,
        warmup_time_total_ms=50.0,
        mean_latency_ms=25.0,
        median_latency_ms=24.5,
        min_latency_ms=22.0,
        max_latency_ms=30.0,
        std_latency_ms=2.5,
        p50_latency_ms=24.5,
        p90_latency_ms=29.0,
        p95_latency_ms=29.8,
        p99_latency_ms=30.0,
        time_to_first_token_ms=10.0,
        latencies_raw_ms=[22.0, 24.0, 24.5, 26.0, 30.0],
    )

    d = metrics.to_dict()
    assert d["mean_latency_ms"] == 25.0
    assert d["p95_latency_ms"] == 29.8
    assert "Mean:" in metrics.summary()
