"""
Model inspection and metadata extraction utilities.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional
import torch
import torch.nn as nn

from inferx.utils.system import bytes_to_mb


@dataclass
class ModelInfo:
    """Structured container for neural model architectural metadata."""
    model_name: str
    architecture: str
    total_parameters: int
    trainable_parameters: int
    num_layers: Optional[int]
    hidden_size: Optional[int]
    num_attention_heads: Optional[int]
    vocab_size: Optional[int]
    dtype: str
    device: str
    estimated_param_memory_mb: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary representation."""
        return asdict(self)

    def summary(self) -> str:
        """Format a human-readable model architectural summary."""
        params_str = f"{self.total_parameters / 1e9:.2f}B" if self.total_parameters >= 1e9 else f"{self.total_parameters / 1e6:.2f}M"
        vocab_str = f"{self.vocab_size:,}" if self.vocab_size else "N/A"
        return (
            f"Model: {self.model_name}\n"
            f"  Architecture: {self.architecture}\n"
            f"  Parameters: {params_str} ({self.total_parameters:,} total)\n"
            f"  Layers: {self.num_layers or 'N/A'} | Hidden Dim: {self.hidden_size or 'N/A'} | Heads: {self.num_attention_heads or 'N/A'}\n"
            f"  Vocab Size: {vocab_str}\n"
            f"  Dtype: {self.dtype} | Device: {self.device}\n"
            f"  Est. Weight Footprint: {self.estimated_param_memory_mb:.1f} MB"
        )


def inspect_model(model: nn.Module, model_name: str, device: str) -> ModelInfo:
    """Extract detailed architectural metadata from a loaded PyTorch model.

    Args:
        model: Loaded PyTorch neural network module.
        model_name: Name or Hugging Face model identifier.
        device: Target execution device string.

    Returns:
        ModelInfo instance populated with inspected properties.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Estimate memory footprint based on tensor element sizes
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    total_bytes += sum(b.numel() * b.element_size() for b in model.buffers())
    est_memory_mb = bytes_to_mb(total_bytes)

    # Extract architecture attributes from HF config if present
    config = getattr(model, "config", None)
    architecture = type(model).__name__
    num_layers = None
    hidden_size = None
    num_heads = None
    vocab_size = None

    if config is not None:
        num_layers = getattr(
            config,
            "num_hidden_layers",
            getattr(config, "n_layer", getattr(config, "num_layers", None)),
        )
        hidden_size = getattr(
            config,
            "hidden_size",
            getattr(config, "n_embd", getattr(config, "d_model", None)),
        )
        num_heads = getattr(
            config,
            "num_attention_heads",
            getattr(config, "n_head", getattr(config, "num_heads", None)),
        )
        vocab_size = getattr(config, "vocab_size", None)

    # Detect active primary dtype
    dtype_str = "unknown"
    for param in model.parameters():
        dtype_str = str(param.dtype).replace("torch.", "")
        break

    return ModelInfo(
        model_name=model_name,
        architecture=architecture,
        total_parameters=total_params,
        trainable_parameters=trainable_params,
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        vocab_size=vocab_size,
        dtype=dtype_str,
        device=device,
        estimated_param_memory_mb=est_memory_mb,
    )
