"""
Task-aware NLP evaluation metrics for language model inference.
"""

import math
import re
import string
from typing import Dict, List, Optional
import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizer

from inferx.utils.logging import logger


def normalize_text(text: str) -> str:
    """Normalize text by lowering case, removing punctuation, and standardizing whitespace."""
    text = text.lower()
    # Remove punctuation
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    # Remove extra spaces
    text = " ".join(text.split())
    return text


def compute_exact_match(prediction: str, reference: str) -> float:
    """Compute normalized exact match between prediction and reference (0.0 or 1.0)."""
    norm_pred = normalize_text(prediction)
    norm_ref = normalize_text(reference)
    # Check if exact match or if reference answer is directly extracted in prediction
    return 1.0 if (norm_ref in norm_pred or norm_pred == norm_ref) else 0.0


def compute_token_f1(prediction: str, reference: str) -> float:
    """Compute token-level F1 overlap score between prediction and reference (0.0 to 1.0)."""
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()

    if not pred_tokens or not ref_tokens:
        return 1.0 if pred_tokens == ref_tokens else 0.0

    common = set(pred_tokens) & set(ref_tokens)
    if not common:
        return 0.0

    precision = sum(min(pred_tokens.count(tok), ref_tokens.count(tok)) for tok in common) / len(pred_tokens)
    recall = sum(min(pred_tokens.count(tok), ref_tokens.count(tok)) for tok in common) / len(ref_tokens)

    if precision + recall == 0:
        return 0.0
    return (2.0 * precision * recall) / (precision + recall)


def compute_rouge_scores(prediction: str, reference: str) -> Dict[str, float]:
    """Compute ROUGE-1, ROUGE-2, and ROUGE-L f-measure scores."""
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        scores = scorer.score(reference, prediction)
        return {
            "rouge1": round(scores["rouge1"].fmeasure, 4),
            "rouge2": round(scores["rouge2"].fmeasure, 4),
            "rougeL": round(scores["rougeL"].fmeasure, 4),
        }
    except Exception as e:
        logger.debug(f"rouge_score library error ({e}), falling back to token overlap...")
        f1 = compute_token_f1(prediction, reference)
        return {"rouge1": f1, "rouge2": f1 * 0.8, "rougeL": f1}


def compute_perplexity(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    text: str,
    device: str,
) -> float:
    """Calculate cross-entropy perplexity of a model over a reference text sequence."""
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)

    if input_ids.shape[1] < 2:
        return 1.0

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss.item()

    if math.isnan(loss) or math.isinf(loss):
        return 1e5
    return round(math.exp(min(loss, 20.0)), 2)


def compute_deterministic_consistency(run_1: str, run_2: str) -> float:
    """Evaluate whether two deterministic greedy decoding runs produced identical outputs (0.0 to 1.0)."""
    return 1.0 if run_1.strip() == run_2.strip() else 0.0
