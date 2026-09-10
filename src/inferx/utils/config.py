"""
Configuration loader and validator for InferX.
"""

from pathlib import Path
from typing import Any, Dict, Optional
import yaml

from inferx.utils.logging import logger


class ConfigError(Exception):
    """Raised when configuration validation or loading fails."""
    pass


def get_project_root() -> Path:
    """Return the absolute path to the InferX project root."""
    # This file is located at <root>/src/inferx/utils/config.py
    return Path(__file__).resolve().parent.parent.parent.parent


def load_yaml(file_path: Path) -> Dict[str, Any]:
    """Safely load and parse a YAML file.

    Args:
        file_path: Path to the YAML file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        ConfigError: If file does not exist or has invalid syntax.
    """
    if not file_path.exists():
        raise ConfigError(f"Configuration file not found: {file_path}")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            return data or {}
    except yaml.YAMLError as e:
        raise ConfigError(f"Failed to parse YAML file {file_path}: {e}") from e
    except Exception as e:
        raise ConfigError(f"Unexpected error loading {file_path}: {e}") from e


def load_models_config(config_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Load model registry configuration."""
    base_dir = config_dir or (get_project_root() / "configs")
    return load_yaml(base_dir / "models.yaml")


def load_benchmark_config(config_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Load benchmark matrix and scoring configuration."""
    base_dir = config_dir or (get_project_root() / "configs")
    return load_yaml(base_dir / "benchmark.yaml")


def load_hardware_config(config_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Load hardware and profiling configuration."""
    base_dir = config_dir or (get_project_root() / "configs")
    return load_yaml(base_dir / "hardware.yaml")


def validate_benchmark_params(
    batch_size: int,
    sequence_length: int,
    max_new_tokens: int,
    precision: str,
    warmup_iterations: int,
    benchmark_iterations: int,
) -> None:
    """Validate benchmark input parameters.

    Raises:
        ValueError: If any benchmark parameter is out of valid bounds.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if sequence_length < 1:
        raise ValueError(f"sequence_length must be >= 1, got {sequence_length}")
    if max_new_tokens < 1:
        raise ValueError(f"max_new_tokens must be >= 1, got {max_new_tokens}")
    valid_precisions = {"fp32", "fp16", "bf16", "int8", "int4"}
    if precision.lower() not in valid_precisions:
        raise ValueError(f"precision '{precision}' is invalid. Supported: {valid_precisions}")
    if warmup_iterations < 0:
        raise ValueError(f"warmup_iterations must be >= 0, got {warmup_iterations}")
    if benchmark_iterations < 1:
        raise ValueError(f"benchmark_iterations must be >= 1, got {benchmark_iterations}")
