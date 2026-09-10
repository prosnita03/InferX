"""
Unit tests for InferX Optimization, Scoring, and Recommendation Engine.
"""

import pytest

from inferx.optimization.optimizer import ParetoOptimizer, is_dominated
from inferx.optimization.recommender import RecommendationEngine
from inferx.optimization.scoring import ObjectiveWeights, ScoringEngine


def test_objective_weights_validation():
    valid = ObjectiveWeights(quality=0.4, throughput=0.3, latency=0.2, memory=0.1)
    valid.validate()

    with pytest.raises(ValueError):
        invalid = ObjectiveWeights(quality=-0.1, throughput=0.3, latency=0.2, memory=0.1)
        invalid.validate()


def test_scoring_engine():
    records = [
        {
            "experiment_id": "cfg_a",
            "mean_latency_ms": 100.0,
            "tokens_per_second": 20.0,
            "peak_memory_mb": 2000.0,
            "quality_score": 80.0,
            "status": "SUCCESS",
        },
        {
            "experiment_id": "cfg_b",
            "mean_latency_ms": 50.0,   # 2x faster
            "tokens_per_second": 40.0,  # 2x throughput
            "peak_memory_mb": 1000.0,  # 2x lower memory
            "quality_score": 82.0,      # higher quality
            "status": "SUCCESS",
        },
    ]

    scorer = ScoringEngine(objective="balanced")
    scored = scorer.score_records(records)

    assert len(scored) == 2
    # cfg_b is strictly superior in every single dimension, so its score must be higher
    assert scored[0]["experiment_id"] == "cfg_b"
    assert scored[0]["score"] > scored[1]["score"]


def test_pareto_dominance():
    # candidate is strictly worse than other
    candidate = {"mean_latency_ms": 100.0, "tokens_per_second": 20.0, "peak_memory_mb": 2000.0, "quality_score": 70.0}
    other = {"mean_latency_ms": 50.0, "tokens_per_second": 40.0, "peak_memory_mb": 1000.0, "quality_score": 80.0}

    assert is_dominated(candidate, other) is True
    assert is_dominated(other, candidate) is False


def test_pareto_frontier_filter():
    records = [
        {"experiment_id": "dom", "mean_latency_ms": 200.0, "tokens_per_second": 10.0, "peak_memory_mb": 3000.0, "quality_score": 50.0, "status": "SUCCESS"},
        {"experiment_id": "fast", "mean_latency_ms": 30.0, "tokens_per_second": 50.0, "peak_memory_mb": 1500.0, "quality_score": 75.0, "status": "SUCCESS"},
        {"experiment_id": "high_qual", "mean_latency_ms": 60.0, "tokens_per_second": 30.0, "peak_memory_mb": 2000.0, "quality_score": 95.0, "status": "SUCCESS"},
    ]

    frontier = ParetoOptimizer.identify_pareto_frontier(records)
    ids = [r["experiment_id"] for r in frontier]

    assert "dom" not in ids
    assert "fast" in ids
    assert "high_qual" in ids


def test_recommendation_engine():
    records = [
        {
            "experiment_id": "baseline",
            "model": "TinyLlama",
            "precision": "fp16",
            "batch_size": 1,
            "sequence_length": 128,
            "kv_cache": True,
            "mean_latency_ms": 100.0,
            "tokens_per_second": 25.0,
            "peak_memory_mb": 2200.0,
            "quality_score": 80.0,
            "status": "SUCCESS",
        },
        {
            "experiment_id": "optimized",
            "model": "TinyLlama",
            "precision": "int8",
            "batch_size": 4,
            "sequence_length": 128,
            "kv_cache": True,
            "mean_latency_ms": 70.0,
            "tokens_per_second": 55.0,
            "peak_memory_mb": 1200.0,
            "quality_score": 79.5,
            "status": "SUCCESS",
        },
    ]

    recommender = RecommendationEngine(objective="high_throughput")
    report = recommender.recommend(records)

    assert report is not None
    assert report.recommended_config["experiment_id"] == "optimized"
    assert report.throughput_change_pct > 0
    assert report.memory_change_pct < 0
    assert len(report.reasons) > 0

    summary_str = report.summary()
    assert "RECOMMENDED CONFIGURATION:" in summary_str
    assert "INT8" in summary_str
