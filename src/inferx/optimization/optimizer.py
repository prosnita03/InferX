"""
Pareto frontier identification and multi-objective trade-off analysis.
"""

from typing import Any, Dict, List


def is_dominated(candidate: Dict[str, Any], other: Dict[str, Any]) -> bool:
    """Check if 'candidate' is dominated by 'other' across inference objectives.

    Objectives:
    - Maximize Quality
    - Maximize Throughput
    - Minimize Latency
    - Minimize Memory
    """
    c_lat = float(candidate.get("mean_latency_ms", 1e9))
    o_lat = float(other.get("mean_latency_ms", 1e9))

    c_thr = float(candidate.get("tokens_per_second", 0.0))
    o_thr = float(other.get("tokens_per_second", 0.0))

    c_mem = float(candidate.get("peak_memory_mb", 1e9))
    o_mem = float(other.get("peak_memory_mb", 1e9))

    c_qua = float(candidate.get("quality_score", 0.0))
    o_qua = float(other.get("quality_score", 0.0))

    # 'other' is at least as good as 'candidate' in all 4 dimensions
    better_or_equal = (
        o_lat <= c_lat and
        o_thr >= c_thr and
        o_mem <= c_mem and
        o_qua >= c_qua
    )

    # 'other' is strictly better in at least one dimension
    strictly_better = (
        o_lat < c_lat or
        o_thr > c_thr or
        o_mem < c_mem or
        o_qua > c_qua
    )

    return better_or_equal and strictly_better


class ParetoOptimizer:
    """Calculates non-dominated Pareto frontier points for inference configurations."""

    @staticmethod
    def identify_pareto_frontier(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter and flag Pareto optimal configurations from benchmark records.

        Args:
            records: List of experiment result dictionaries.

        Returns:
            List of non-dominated Pareto optimal records.
        """
        valid = [r for r in records if r.get("status") == "SUCCESS"]
        if not valid:
            return []

        pareto_front: List[Dict[str, Any]] = []

        for candidate in valid:
            dominated = False
            for other in valid:
                if candidate is other:
                    continue
                if is_dominated(candidate, other):
                    dominated = True
                    break

            candidate["is_pareto_optimal"] = not dominated
            if not dominated:
                pareto_front.append(candidate)

        return pareto_front
