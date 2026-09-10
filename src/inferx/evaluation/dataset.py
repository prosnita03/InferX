"""
Dataset management for InferX multi-task evaluation pipelines.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from inferx.utils.config import get_project_root
from inferx.utils.logging import logger


class EvaluationDataset:
    """Manages benchmark prompts across question answering, factual recall, reasoning, and summarization."""

    def __init__(self, dataset_path: Optional[Path] = None):
        """Initialize evaluation dataset.

        Args:
            dataset_path: Custom JSON dataset path. Defaults to data/eval_dataset.json.
        """
        self.dataset_path = dataset_path or (get_project_root() / "data" / "eval_dataset.json")
        self.samples = self._load()

    def _load(self) -> List[Dict[str, Any]]:
        """Load and validate dataset items."""
        if not self.dataset_path.exists():
            logger.warning(f"Dataset path {self.dataset_path} not found. Returning empty dataset.")
            return []

        try:
            with open(self.dataset_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if not isinstance(data, list):
                    raise ValueError("Dataset JSON must be a list of sample objects.")
                return data
        except Exception as e:
            logger.error(f"Failed to load evaluation dataset: {e}")
            return []

    def get_by_task(self, task: str) -> List[Dict[str, Any]]:
        """Filter dataset samples by specific task category."""
        return [s for s in self.samples if s.get("task") == task]

    def get_all(self) -> List[Dict[str, Any]]:
        """Return full dataset collection."""
        return self.samples

    def get_demo_samples(self, count: int = 3) -> List[Dict[str, Any]]:
        """Return a compact subset of representative samples for rapid demonstration."""
        return self.samples[:count]
