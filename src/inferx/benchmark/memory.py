"""
Memory profiling and tracking utilities for GPU and CPU backends.
"""

from dataclasses import asdict, dataclass
import gc
import os
from typing import Any, Dict, Optional
import psutil
import torch

from inferx.utils.logging import logger
from inferx.utils.system import bytes_to_mb


@dataclass
class MemorySnapshot:
    """Represents memory consumption at a specific execution instant."""
    device: str
    baseline_mb: float
    peak_mb: float
    delta_mb: float
    reserved_mb: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert snapshot to dictionary."""
        return asdict(self)

    def summary(self) -> str:
        """Format human-readable memory summary."""
        dev_label = "GPU VRAM" if "cuda" in self.device else ("MPS Memory" if "mps" in self.device else "CPU RAM (RSS)")
        res_str = f" | Reserved: {self.reserved_mb:.1f} MB" if self.reserved_mb is not None else ""
        return (
            f"[{dev_label}] Baseline: {self.baseline_mb:.1f} MB | "
            f"Peak: {self.peak_mb:.1f} MB | Delta: {self.delta_mb:+.1f} MB{res_str}"
        )


class MemoryTracker:
    """Context manager and tracker for isolating peak memory utilization during inference."""

    def __init__(self, device: str):
        """Initialize memory tracker for the specified target device.

        Args:
            device: Device string ('cuda', 'mps', 'cpu').
        """
        self.device = str(device).lower().split(":")[0]
        self.process = psutil.Process(os.getpid())
        self.baseline_bytes = 0
        self.peak_bytes = 0
        self.reserved_bytes: Optional[int] = None

    def _get_current_bytes(self) -> int:
        """Query current instantaneous device or process memory allocation."""
        if self.device == "cuda" and torch.cuda.is_available():
            return torch.cuda.memory_allocated()
        elif self.device == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "current_allocated_memory"):
            try:
                return torch.mps.current_allocated_memory()
            except Exception:
                return self.process.memory_info().rss
        else:
            return self.process.memory_info().rss

    def __enter__(self) -> "MemoryTracker":
        """Start tracking: force garbage collection and snapshot baseline."""
        gc.collect()
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            self.baseline_bytes = torch.cuda.memory_allocated()
        else:
            self.baseline_bytes = self._get_current_bytes()

        self.peak_bytes = self.baseline_bytes
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Conclude tracking and capture maximum memory allocation."""
        if self.device == "cuda" and torch.cuda.is_available():
            self.peak_bytes = torch.cuda.max_memory_allocated()
            self.reserved_bytes = torch.cuda.memory_reserved()
        else:
            # For CPU/MPS, capture current after operation (peak estimation)
            current = self._get_current_bytes()
            self.peak_bytes = max(self.baseline_bytes, current)

    def get_snapshot(self) -> MemorySnapshot:
        """Retrieve structured memory snapshot from the completed tracking interval."""
        baseline_mb = bytes_to_mb(self.baseline_bytes)
        peak_mb = bytes_to_mb(self.peak_bytes)
        delta_mb = round(peak_mb - baseline_mb, 2)
        reserved_mb = bytes_to_mb(self.reserved_bytes) if self.reserved_bytes is not None else None

        return MemorySnapshot(
            device=self.device,
            baseline_mb=baseline_mb,
            peak_mb=peak_mb,
            delta_mb=delta_mb,
            reserved_mb=reserved_mb,
        )
