"""
Quantization dispatch and configuration manager for InferX.
"""

from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn

from inferx.utils.logging import logger
from inferx.utils.system import HardwareNotSupportedError


def is_bitsandbytes_available() -> bool:
    """Check if bitsandbytes library is installed and loadable."""
    try:
        import bitsandbytes  # noqa: F401
        return True
    except (ImportError, Exception):
        return False


def get_precision_bytes_per_param(precision: str) -> float:
    """Return theoretical bytes required per parameter for a given precision."""
    mapping = {
        "fp32": 4.0,
        "fp16": 2.0,
        "bf16": 2.0,
        "int8": 1.0,
        "int4": 0.5,
    }
    prec = precision.lower()
    if prec not in mapping:
        raise ValueError(f"Unknown precision: {precision}. Supported: {list(mapping.keys())}")
    return mapping[prec]


class QuantizationManager:
    """Manages quantization configurations and hardware compatibility checks."""

    @staticmethod
    def get_torch_dtype_and_kwargs(
        precision: str,
        device: str,
    ) -> Tuple[torch.dtype, Dict[str, Any]]:
        """Validate precision against active hardware and generate Hugging Face kwargs.

        Args:
            precision: Requested precision ('fp32', 'fp16', 'bf16', 'int8', 'int4').
            device: Target device ('cuda', 'mps', 'cpu').

        Returns:
            Tuple of (primary torch.dtype, hf_model_kwargs dictionary).

        Raises:
            HardwareNotSupportedError: If requested precision cannot be satisfied.
        """
        prec = precision.lower()
        dev = device.lower().split(":")[0]

        if prec == "fp32":
            return torch.float32, {}

        if prec in ("fp16", "float16"):
            if dev == "cpu":
                logger.warning("FP16 on CPU may run slowly or with emulation. FP32 is recommended for CPU.")
                return torch.float32, {}
            return torch.float16, {}

        if prec in ("bf16", "bfloat16"):
            if dev == "cuda" and not torch.cuda.is_bf16_supported():
                raise HardwareNotSupportedError("Current CUDA device does not support bfloat16.")
            return torch.bfloat16, {}

        if prec == "int8":
            if dev == "cuda":
                if not is_bitsandbytes_available():
                    raise HardwareNotSupportedError(
                        "bitsandbytes is not installed or available on this system. "
                        "Install with `pip install bitsandbytes` on Linux/Windows CUDA systems."
                    )
                try:
                    from transformers import BitsAndBytesConfig
                    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
                    return torch.float16, {"quantization_config": bnb_config}
                except Exception as e:
                    raise HardwareNotSupportedError(f"Failed to initialize 8-bit quantization: {e}") from e
            elif dev == "cpu":
                # On CPU, return fp32 for initial load; post-load dynamic int8 will be applied
                return torch.float32, {"_inferx_cpu_int8": True}
            else:
                raise HardwareNotSupportedError(
                    f"INT8 quantization is not natively supported on device '{device}'. "
                    f"Requires CUDA with bitsandbytes or CPU dynamic quantization."
                )

        if prec == "int4":
            if dev != "cuda":
                raise HardwareNotSupportedError(
                    f"INT4 NF4/FP4 quantization requires an NVIDIA GPU with CUDA. "
                    f"Current device is '{device}'."
                )
            if not is_bitsandbytes_available():
                raise HardwareNotSupportedError(
                    "bitsandbytes is not installed. Required for INT4 quantization."
                )
            try:
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                return torch.float16, {"quantization_config": bnb_config}
            except Exception as e:
                raise HardwareNotSupportedError(f"Failed to initialize 4-bit quantization: {e}") from e

        raise ValueError(f"Unsupported precision: {precision}")

    @staticmethod
    def apply_cpu_dynamic_int8(model: nn.Module) -> nn.Module:
        """Apply PyTorch native dynamic INT8 quantization to linear layers for CPU execution.

        Args:
            model: PyTorch model loaded in FP32.

        Returns:
            Dynamically quantized model.
        """
        try:
            logger.info("Applying PyTorch native dynamic INT8 quantization to nn.Linear layers for CPU...")
            quantized_model = torch.ao.quantization.quantize_dynamic(
                model,
                {nn.Linear},
                dtype=torch.qint8,
            )
            return quantized_model
        except Exception as e:
            logger.warning(f"Could not apply dynamic INT8 quantization on this architecture: {e}. Keeping FP32.")
            return model
