"""
System, timing, memory, and reproducibility utilities for InferX.
"""

import os
import random
import time
from typing import Optional, Union
import numpy as np
import torch

from inferx.utils.logging import logger


class InferXError(Exception):
    """Base exception class for all InferX errors."""
    pass


class HardwareNotSupportedError(InferXError):
    """Raised when an operation is unsupported on current hardware."""
    pass


class OutOfMemoryError(InferXError):
    """Raised when an allocation or benchmark exceeds available memory."""
    pass


def set_seed(seed: int = 42) -> None:
    """Set global random seeds across libraries for strict reproducibility.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed"):
        try:
            torch.mps.manual_seed(seed)
        except Exception:
            pass


def sync_device(device: Union[str, torch.device]) -> None:
    """Synchronize compute device execution streams before timing operations.

    Crucial for accurate GPU/accelerator latency measurement without timing
    asynchronous host kernel queues.

    Args:
        device: Target torch device or string representation.
    """
    device_type = str(device).split(":")[0].lower()
    if device_type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device_type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        try:
            torch.mps.synchronize()
        except Exception:
            pass


def bytes_to_mb(byte_count: Union[int, float]) -> float:
    """Convert raw byte count to megabytes (MiB)."""
    return round(byte_count / (1024.0 * 1024.0), 2)


def bytes_to_gb(byte_count: Union[int, float]) -> float:
    """Convert raw byte count to gigabytes (GiB)."""
    return round(byte_count / (1024.0 * 1024.0 * 1024.0), 3)


def format_latency(ms: float) -> str:
    """Format milliseconds into a human-readable duration."""
    if ms < 1.0:
        return f"{ms * 1000.0:.1f} µs"
    if ms < 1000.0:
        return f"{ms:.2f} ms"
    return f"{ms / 1000.0:.2f} s"


def format_throughput(tokens_per_sec: float) -> str:
    """Format throughput into human-readable token generation rate."""
    return f"{tokens_per_sec:.2f} tok/s"


def calculate_pct_diff(baseline: float, current: float) -> float:
    """Calculate percentage change relative to baseline.

    Positive means increase, negative means decrease.
    Returns 0.0 if baseline is 0.
    """
    if baseline == 0:
        return 0.0
    return round(((current - baseline) / baseline) * 100.0, 2)
