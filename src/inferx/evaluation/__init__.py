"""
Model quality and task-aware evaluation modules for InferX.
"""

from inferx.evaluation.dataset import EvaluationDataset
from inferx.evaluation.metrics import (
    compute_exact_match,
    compute_token_f1,
    compute_rouge_scores,
    compute_perplexity,
    compute_deterministic_consistency,
)
from inferx.evaluation.evaluator import Evaluator, EvaluationReport, SampleEvaluationResult

__all__ = [
    "EvaluationDataset",
    "compute_exact_match",
    "compute_token_f1",
    "compute_rouge_scores",
    "compute_perplexity",
    "compute_deterministic_consistency",
    "Evaluator",
    "EvaluationReport",
    "SampleEvaluationResult",
]
