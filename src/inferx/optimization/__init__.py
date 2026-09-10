"""
Optimization, multi-objective scoring, and Pareto recommendation modules for InferX.
"""

from inferx.optimization.scoring import ScoringEngine, ObjectiveWeights, OBJECTIVE_PRESETS
from inferx.optimization.optimizer import ParetoOptimizer, is_dominated
from inferx.optimization.recommender import RecommendationEngine, RecommendationReport

__all__ = [
    "ScoringEngine",
    "ObjectiveWeights",
    "OBJECTIVE_PRESETS",
    "ParetoOptimizer",
    "is_dominated",
    "RecommendationEngine",
    "RecommendationReport",
]
