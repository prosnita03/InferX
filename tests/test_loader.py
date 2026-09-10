"""
Unit tests for InferX Model Loader and Model Info extraction.
"""

from unittest.mock import MagicMock
import pytest
import torch
import torch.nn as nn

from inferx.models.loader import ModelLoader
from inferx.models.model_info import ModelInfo, inspect_model
from inferx.models.quantization import QuantizationManager, get_precision_bytes_per_param
from inferx.utils.system import HardwareNotSupportedError


class DummyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(32, 32)
        self.k_proj = nn.Linear(32, 32)
        self.v_proj = nn.Linear(32, 32)

    def forward(self, x):
        return self.v_proj(x)


class DummyDecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = DummyAttention()
        self.mlp = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 32))

    def forward(self, x):
        return self.mlp(self.self_attn(x))


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(100, 32)
        self.layers = nn.ModuleList([DummyDecoderLayer() for _ in range(2)])
        self.lm_head = nn.Linear(32, 100)

        # Mock config
        self.config = MagicMock()
        self.config.num_hidden_layers = 2
        self.config.hidden_size = 32
        self.config.num_attention_heads = 4
        self.config.vocab_size = 100

    def forward(self, input_ids, **kwargs):
        x = self.embed_tokens(input_ids)
        for l in self.layers:
            x = l(x)
        logits = self.lm_head(x)
        MockOutput = MagicMock()
        MockOutput.logits = logits
        MockOutput.past_key_values = None
        return MockOutput


def test_model_info_inspection():
    model = DummyModel()
    info = inspect_model(model, model_name="dummy-test-model", device="cpu")

    assert info.model_name == "dummy-test-model"
    assert info.total_parameters > 0
    assert info.trainable_parameters == info.total_parameters
    assert info.num_layers == 2
    assert info.hidden_size == 32
    assert info.num_attention_heads == 4
    assert info.vocab_size == 100
    assert info.estimated_param_memory_mb > 0.0

    summary_text = info.summary()
    assert "dummy-test-model" in summary_text
    assert "Parameters:" in summary_text


def test_precision_bytes_mapping():
    assert get_precision_bytes_per_param("fp32") == 4.0
    assert get_precision_bytes_per_param("fp16") == 2.0
    assert get_precision_bytes_per_param("int8") == 1.0
    assert get_precision_bytes_per_param("int4") == 0.5

    with pytest.raises(ValueError):
        get_precision_bytes_per_param("int2")


def test_quantization_manager_dispatch():
    # FP32 on CPU
    dtype, kwargs = QuantizationManager.get_torch_dtype_and_kwargs("fp32", "cpu")
    assert dtype == torch.float32

    # INT4 on CPU must raise HardwareNotSupportedError
    with pytest.raises(HardwareNotSupportedError):
        QuantizationManager.get_torch_dtype_and_kwargs("int4", "cpu")


def test_model_loader_device_detection():
    loader = ModelLoader()
    assert loader.device in ["cuda", "mps", "cpu"]
