"""
Unit tests for InferX Task-Aware Evaluation Metrics.
"""

import pytest

from inferx.evaluation.dataset import EvaluationDataset
from inferx.evaluation.metrics import (
    compute_exact_match,
    compute_token_f1,
    compute_rouge_scores,
    compute_deterministic_consistency,
    normalize_text,
)


def test_normalize_text():
    raw = "  Hello, World! This is a TEST.  "
    norm = normalize_text(raw)
    assert norm == "hello world this is a test"


def test_exact_match():
    assert compute_exact_match("Paris", "Paris") == 1.0
    assert compute_exact_match("paris.", "Paris") == 1.0
    assert compute_exact_match("The capital is Paris!", "Paris") == 1.0
    assert compute_exact_match("London", "Paris") == 0.0


def test_token_f1():
    # Complete match
    assert compute_token_f1("Graphics Processing Unit", "Graphics Processing Unit") == 1.0
    # Partial match
    score = compute_token_f1("Graphics Processing", "Graphics Processing Unit")
    assert 0.0 < score < 1.0
    # Zero match
    assert compute_token_f1("Central Logic", "Graphics Processing Unit") == 0.0


def test_rouge_scores():
    pred = "Machine learning learns patterns from data automatically."
    ref = "Machine learning is an automated pattern recognition method from data."
    scores = compute_rouge_scores(pred, ref)

    assert "rouge1" in scores
    assert "rouge2" in scores
    assert "rougeL" in scores
    assert scores["rougeL"] > 0.0


def test_deterministic_consistency():
    assert compute_deterministic_consistency("output token stream", "output token stream") == 1.0
    assert compute_deterministic_consistency("output token stream A", "output token stream B") == 0.0


def test_evaluation_dataset_loading():
    dataset = EvaluationDataset()
    samples = dataset.get_all()
    assert len(samples) > 0

    qa_samples = dataset.get_by_task("question_answering")
    assert len(qa_samples) > 0
    assert qa_samples[0]["task"] == "question_answering"

    demo_samples = dataset.get_demo_samples(2)
    assert len(demo_samples) == 2
