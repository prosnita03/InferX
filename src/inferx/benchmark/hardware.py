"""
Hardware detection and environment profiling for InferX.
"""

from dataclasses import asdict, dataclass
import platform
import sys
from typing import Any, Dict, Optional
import psutil
import torch

from inferx.utils.logging import logger
from inferx.utils.system import bytes_to_gb


@dataclass
class HardwareProfile:
    """System hardware and runtime environment profile."""
    os: str
    os_release: str
    python_version: str
    pytorch_version: str
    cuda_available: bool
    cuda_version: Optional[str]
    mps_available: bool
    device_count: int
    gpu_name: Optional[str]
    gpu_total_memory_gb: Optional[float]
    cpu_model: str
    cpu_cores_physical: int
    cpu_cores_logical: int
    system_ram_gb: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert hardware profile to dictionary."""
        return asdict(self)

    def summary(self) -> str:
        """Format an executive hardware environment summary."""
        gpu_str = (
            f"{self.gpu_name} ({self.gpu_total_memory_gb:.1f} GB VRAM)"
            if self.cuda_available and self.gpu_name
            else ("Apple Silicon (Metal/MPS)" if self.mps_available else "None (CPU only)")
        )
        return (
            f"Hardware Environment:\n"
            f"  OS:               {self.os} ({self.os_release})\n"
            f"  Python:           {self.python_version}\n"
            f"  PyTorch:          {self.pytorch_version}\n"
            f"  CUDA Available:   {self.cuda_available} (CUDA {self.cuda_version or 'N/A'})\n"
            f"  MPS Available:    {self.mps_available}\n"
            f"  Primary Compute:  {gpu_str}\n"
            f"  CPU:              {self.cpu_model} ({self.cpu_cores_physical} physical, {self.cpu_cores_logical} logical cores)\n"
            f"  System RAM:       {self.system_ram_gb:.1f} GB"
        )


class HardwareProfiler:
    """Collects comprehensive operating system and accelerator hardware characteristics."""

    @staticmethod
    def get_cpu_model_name() -> str:
        """Determine human-readable CPU brand string."""
        try:
            if platform.system() == "Darwin":
                import subprocess
                cmd = ["sysctl", "-n", "machdep.cpu.brand_string"]
                res = subprocess.check_output(cmd).decode().strip()
                if res:
                    return res
            elif platform.system() == "Linux":
                with open("/proc/cpuinfo", "r") as f:
                    for line in f:
                        if "model name" in line:
                            return line.split(":", 1)[1].strip()
        except Exception:
            pass
        return platform.processor() or "Unknown CPU"

    @classmethod
    def profile(cls) -> HardwareProfile:
        """Profile and return current host hardware specifications.

        Returns:
            HardwareProfile dataclass populated with host metrics.
        """
        cuda_avail = torch.cuda.is_available()
        cuda_ver = torch.version.cuda if cuda_avail else None
        mps_avail = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

        gpu_name = None
        gpu_vram = None
        device_count = 0

        if cuda_avail:
            device_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            gpu_vram = bytes_to_gb(torch.cuda.get_device_properties(0).total_memory)

        ram = psutil.virtual_memory()
        physical_cores = psutil.cpu_count(logical=False) or 1
        logical_cores = psutil.cpu_count(logical=True) or 1

        profile = HardwareProfile(
            os=platform.system(),
            os_release=platform.release(),
            python_version=sys.version.split()[0],
            pytorch_version=torch.__version__,
            cuda_available=cuda_avail,
            cuda_version=cuda_ver,
            mps_available=mps_avail,
            device_count=device_count,
            gpu_name=gpu_name,
            gpu_total_memory_gb=gpu_vram,
            cpu_model=cls.get_cpu_model_name(),
            cpu_cores_physical=physical_cores,
            cpu_cores_logical=logical_cores,
            system_ram_gb=bytes_to_gb(ram.total),
        )
        return profile
