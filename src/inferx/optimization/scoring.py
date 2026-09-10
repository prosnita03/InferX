"""
Multi-objective optimization scoring engine for InferX.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np

from inferx.utils.logging import logger


@dataclass
class ObjectiveWeights:
    """Weights assigned to competing optimization dimensions (must sum to ~1.0)."""
    quality: float = 0.35
    throughput: float = 0.25
    latency: float = 0.25
    memory: float = 0.15

    def validate(self) -> None:
        """Ensure all weights are non-negative."""
        if any(w < 0 for w in [self.quality, self.throughput, self.latency, self.memory]):
            raise ValueError("All scoring weights must be non-negative.")
        total = self.quality + self.throughput + self.latency + self.memory
        if total <= 0:
            raise ValueError("Sum of scoring weights must be greater than 0.")


OBJECTIVE_PRESETS: Dict[str, ObjectiveWeights] = {
    "balanced": ObjectiveWeights(quality=0.35, throughput=0.25, latency=0.25, memory=0.15),
    "low_latency": ObjectiveWeights(quality=0.20, throughput=0.10, latency=0.60, memory=0.10),
    "high_throughput": ObjectiveWeights(quality=0.20, throughput=0.60, latency=0.10, memory=0.10),
    "low_memory": ObjectiveWeights(quality=0.20, throughput=0.10, latency=0.10, memory=0.60),
    "high_quality": ObjectiveWeights(quality=0.70, throughput=0.10, latency=0.10, memory=0.10),
}


class ScoringEngine:
    """Computes normalized composite scores for candidate inference configurations."""

    def __init__(self, objective: str = "balanced", custom_weights: Optional[ObjectiveWeights] = None):
        """Initialize scoring engine with a predefined objective or custom weights.

        Args:
            objective: One of 'balanced', 'low_latency', 'high_throughput', 'low_memory', 'high_quality'.
            custom_weights: Optional ObjectiveWeights instance overriding the preset.
        """
        if custom_weights:
            custom_weights.validate()
            self.weights = custom_weights
            self.objective_name = "custom"
        else:
            obj_key = objective.lower().replace(" ", "_")
            if obj_key not in OBJECTIVE_PRESETS:
                logger.warning(f"Unknown objective '{objective}', falling back to 'balanced'.")
                obj_key = "balanced"
            self.weights = OBJECTIVE_PRESETS[obj_key]
            self.objective_name = obj_key

    @staticmethod
    def _min_max_normalize(values: List[float], invert: bool = False) -> List[float]:
        """Normalize a series of values to [0.0, 1.0].

        Args:
            values: List of numeric values.
            invert: If True, smaller is better (e.g. latency, memory).
                   If False, larger is better (e.g. throughput, quality).
        """
        arr = np.array(values, dtype=float)
        v_min, v_max = float(np.min(arr)), float(np.max(arr))

        if v_max == v_min:
            return [1.0 if not invert else 1.0 for _ in values]

        norm = (arr - v_min) / (v_max - v_min)
        if invert:
            norm = 1.0 - norm
        return [float(x) for x in norm]

    def score_records(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Score and rank experiment records according to active objective weights.

        Higher score is always better (range approximately 0.0 to 100.0).

        Args:
            records: List of experiment result dictionaries.

        Returns:
            List of experiment records enriched with 'score' and normalized components.
        """
        if not records:
            return []

        # Filter out records that crashed/OOMed
        valid_records = [r for r in records if r.get("status") == "SUCCESS"]
        if not valid_records:
            return records

        latencies = [float(r.get("mean_latency_ms", 1.0)) for r in valid_records]
        throughputs = [float(r.get("tokens_per_second", 0.0)) for r in valid_records]
        memories = [float(r.get("peak_memory_mb", 1.0)) for r in valid_records]
        qualities = [float(r.get("quality_score", 50.0)) for r in valid_records]

        # Invert latency and memory so that lower values yield higher normalized scores
        norm_lat = self._min_max_normalize(latencies, invert=True)
        norm_thr = self._min_max_normalize(throughputs, invert=False)
        norm_mem = self._min_max_normalize(memories, invert=True)
        norm_qua = self._min_max_normalize(qualities, invert=False)

        total_weight = self.weights.quality + self.weights.throughput + self.weights.latency + self.weights.memory

        for idx, r in enumerate(valid_records):
            composite = (
                self.weights.quality * norm_qua[idx]
                + self.weights.throughput * norm_thr[idx]
                + self.weights.latency * norm_lat[idx]
                + self.weights.memory * norm_mem[idx]
            ) / total_weight

            r["score"] = round(composite * 100.0, 2)
            r["norm_latency"] = round(norm_lat[idx], 3)
            r["norm_throughput"] = round(norm_thr[idx], 3)
            r["norm_memory"] = round(norm_mem[idx], 3)
            r["norm_quality"] = round(norm_qua[idx], 3)

        # Sort valid records descending by score
        valid_records.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return valid_records
