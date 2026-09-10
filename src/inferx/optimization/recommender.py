"""
Automated inference optimization recommendation engine for InferX.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from inferx.optimization.optimizer import ParetoOptimizer
from inferx.optimization.scoring import ScoringEngine
from inferx.utils.logging import logger
from inferx.utils.system import calculate_pct_diff, format_latency, format_throughput


@dataclass
class RecommendationReport:
    """Consolidated optimization recommendation with baseline comparison."""
    objective: str
    recommended_config: Dict[str, Any]
    baseline_config: Dict[str, Any]
    latency_change_pct: float
    throughput_change_pct: float
    memory_change_pct: float
    quality_change_pct: float
    reasons: List[str]
    pareto_optimal_count: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert recommendation report to dictionary."""
        return asdict(self)

    def summary(self) -> str:
        """Format an executive human-readable recommendation summary."""
        rec = self.recommended_config
        base = self.baseline_config

        reasons_str = "\n".join(f"    - {r}" for r in self.reasons)

        return (
            f"==================================================\n"
            f"InferX Optimization Recommendation\n"
            f"Objective: {self.objective.upper()}\n"
            f"==================================================\n\n"
            f"RECOMMENDED CONFIGURATION:\n"
            f"  Model:           {rec.get('model', 'N/A')}\n"
            f"  Precision:       {rec.get('precision', 'N/A').upper()}\n"
            f"  Batch Size:      {rec.get('batch_size', 1)}\n"
            f"  Sequence Length: {rec.get('sequence_length', 'N/A')}\n"
            f"  KV Cache:        {'Enabled' if rec.get('kv_cache', True) else 'Disabled'}\n"
            f"  Composite Score: {rec.get('score', 0.0):.1f}/100.0\n\n"
            f"KEY ADVANTAGES:\n{reasons_str}\n\n"
            f"COMPARISON AGAINST BASELINE ({base.get('precision', 'N/A').upper()} / Batch {base.get('batch_size', 1)}):\n"
            f"  Latency:    {format_latency(rec.get('mean_latency_ms', 0))} vs {format_latency(base.get('mean_latency_ms', 0))} ({self.latency_change_pct:+.1f}%)\n"
            f"  Throughput: {format_throughput(rec.get('tokens_per_second', 0))} vs {format_throughput(base.get('tokens_per_second', 0))} ({self.throughput_change_pct:+.1f}%)\n"
            f"  Peak Mem:   {rec.get('peak_memory_mb', 0):.1f} MB vs {base.get('peak_memory_mb', 0):.1f} MB ({self.memory_change_pct:+.1f}%)\n"
            f"  Quality:    {rec.get('quality_score', 0):.1f} vs {base.get('quality_score', 0):.1f} ({self.quality_change_pct:+.1f}%)\n"
            f"=================================================="
        )


class RecommendationEngine:
    """Evaluates benchmark history and produces data-backed configuration recommendations."""

    def __init__(self, objective: str = "balanced"):
        """Initialize RecommendationEngine with a specific optimization objective."""
        self.objective = objective
        self.scoring_engine = ScoringEngine(objective=objective)

    def recommend(self, records: List[Dict[str, Any]]) -> Optional[RecommendationReport]:
        """Identify optimal configuration from records and formulate comparative report.

        Args:
            records: Benchmark run records.

        Returns:
            RecommendationReport or None if no valid records exist.
        """
        if not records:
            logger.warning("Cannot generate recommendation: No records provided.")
            return None

        # 1. Score records
        scored_records = self.scoring_engine.score_records(records)
        if not scored_records:
            logger.warning("No successfully completed records found to score.")
            return None

        # 2. Identify Pareto frontier
        pareto_records = ParetoOptimizer.identify_pareto_frontier(scored_records)

        # 3. Best configuration according to objective score
        recommended = scored_records[0]

        # 4. Identify baseline (prefer batch_size=1 and fp16 or fp32)
        baseline = None
        for r in scored_records:
            if r.get("batch_size") == 1 and r.get("precision") in ("fp16", "fp32") and r.get("kv_cache") is True:
                baseline = r
                break
        if not baseline:
            # Fallback to the first batch_size=1 record or lowest batch size record
            single_batch = [r for r in scored_records if r.get("batch_size") == 1]
            baseline = single_batch[0] if single_batch else scored_records[-1]

        # 5. Compute percentage differentials
        lat_diff = calculate_pct_diff(
            float(baseline.get("mean_latency_ms", 1.0)),
            float(recommended.get("mean_latency_ms", 1.0)),
        )
        thr_diff = calculate_pct_diff(
            float(baseline.get("tokens_per_second", 1.0)),
            float(recommended.get("tokens_per_second", 1.0)),
        )
        mem_diff = calculate_pct_diff(
            float(baseline.get("peak_memory_mb", 1.0)),
            float(recommended.get("peak_memory_mb", 1.0)),
        )
        qua_diff = calculate_pct_diff(
            float(baseline.get("quality_score", 1.0)),
            float(recommended.get("quality_score", 1.0)),
        )

        # 6. Synthesize explainable justification reasons
        reasons = []
        if mem_diff < -5.0:
            reasons.append(f"Reduced memory footprint by {abs(mem_diff):.1f}% relative to baseline.")
        elif mem_diff <= 5.0:
            reasons.append("Maintains conservative memory footprint within safe hardware boundaries.")
        else:
            reasons.append(f"Acceptable memory trade-off (+{mem_diff:.1f}%) for throughput gains.")

        if thr_diff > 5.0:
            reasons.append(f"Boosted throughput by {thr_diff:.1f}% for higher concurrent serving capacity.")

        if lat_diff < -5.0:
            reasons.append(f"Cut inference latency by {abs(lat_diff):.1f}% for lower user wait time.")
        elif lat_diff <= 15.0:
            reasons.append("Maintains low, interactive response latency.")

        if qua_diff >= -2.0:
            reasons.append("Preserves full model fidelity with negligible or zero degradation.")
        else:
            reasons.append(f"Slight quality trade-off ({qua_diff:.1f}%) within acceptable operational thresholds.")

        if recommended.get("is_pareto_optimal", False):
            reasons.append("Guaranteed Pareto-optimal: no alternative configuration achieves strictly superior metrics in all dimensions.")

        report = RecommendationReport(
            objective=self.objective,
            recommended_config=recommended,
            baseline_config=baseline,
            latency_change_pct=lat_diff,
            throughput_change_pct=thr_diff,
            memory_change_pct=mem_diff,
            quality_change_pct=qua_diff,
            reasons=reasons,
            pareto_optimal_count=len(pareto_records),
        )

        logger.info(f"\n{report.summary()}")
        return report
