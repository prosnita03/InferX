"""
Model loading and device dispatch engine for InferX.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from inferx.models.model_info import ModelInfo, inspect_model
from inferx.models.quantization import QuantizationManager
from inferx.utils.logging import logger
from inferx.utils.system import HardwareNotSupportedError, InferXError


class ModelLoader:
    """Robust loader for Hugging Face causal language models and tokenizers."""

    def __init__(self, device: Optional[str] = None):
        """Initialize ModelLoader with target device or automatic detection.

        Args:
            device: Explicit device string ('cuda', 'cuda:0', 'mps', 'cpu') or None for auto-detect.
        """
        self.device = device or self.detect_default_device()
        logger.info(f"ModelLoader initialized with target device: {self.device}")

    @staticmethod
    def detect_default_device() -> str:
        """Detect the highest-priority compute accelerator available."""
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def load(
        self,
        model_name: str,
        precision: str = "fp16",
        device_map: Optional[str] = None,
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer, ModelInfo]:
        """Load a Hugging Face causal language model, tokenizer, and inspect architecture.

        Args:
            model_name: Hugging Face repo ID or local checkpoint directory.
            precision: Precision string ('fp32', 'fp16', 'bf16', 'int8', 'int4').
            device_map: Optional Hugging Face device map override (e.g. 'auto').

        Returns:
            Tuple of (model, tokenizer, model_info).

        Raises:
            HardwareNotSupportedError: If precision is unsupported on the device.
            InferXError: If model or tokenizer fails to load.
        """
        logger.info(f"Loading model '{model_name}' with precision={precision} on device={self.device}...")

        # 1. Load Tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
            )
            # Ensure padding token exists for batch inference
            if tokenizer.pad_token is None:
                if tokenizer.eos_token is not None:
                    tokenizer.pad_token = tokenizer.eos_token
                    logger.debug("Set tokenizer.pad_token to tokenizer.eos_token.")
                else:
                    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                    logger.debug("Added default [PAD] token to tokenizer.")

            # Causal LM batch generation requires left padding so newly generated tokens
            # are aligned across all sequences in the batch
            tokenizer.padding_side = "left"

        except Exception as e:
            raise InferXError(f"Failed to load tokenizer for '{model_name}': {e}") from e

        # 2. Configure precision & kwargs
        torch_dtype, model_kwargs = QuantizationManager.get_torch_dtype_and_kwargs(
            precision=precision,
            device=self.device,
        )
        is_cpu_int8 = model_kwargs.pop("_inferx_cpu_int8", False)

        # 3. Load Model
        try:
            # If 8-bit or 4-bit bnb is used, device_map='auto' is required by HF
            if "quantization_config" in model_kwargs:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch_dtype,
                    device_map=device_map or "auto",
                    trust_remote_code=True,
                    **model_kwargs,
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch_dtype,
                    trust_remote_code=True,
                    **model_kwargs,
                )
                if self.device != "cpu":
                    model = model.to(self.device)

            # Apply CPU dynamic int8 if requested
            if is_cpu_int8 and self.device == "cpu":
                model = QuantizationManager.apply_cpu_dynamic_int8(model)

            # Set model to evaluation mode
            model.eval()

        except HardwareNotSupportedError:
            raise
        except Exception as e:
            raise InferXError(f"Failed to load model '{model_name}': {e}") from e

        # 4. Extract Model Architecture Information
        model_info = inspect_model(model, model_name=model_name, device=self.device)
        logger.info(f"Model loaded successfully:\n{model_info.summary()}")

        return model, tokenizer, model_info
