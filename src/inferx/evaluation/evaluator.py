"""
Comprehensive multi-task evaluation orchestrator for InferX.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from inferx.evaluation.dataset import EvaluationDataset
from inferx.evaluation.metrics import (
    compute_exact_match,
    compute_token_f1,
    compute_rouge_scores,
    compute_deterministic_consistency,
)
from inferx.inference.engine import InferenceEngine
from inferx.utils.logging import logger


@dataclass
class SampleEvaluationResult:
    """Individual sample evaluation outcome."""
    sample_id: str
    task: str
    prompt: str
    reference: str
    prediction: str
    score: float
    metric_name: str


@dataclass
class EvaluationReport:
    """Consolidated model quality assessment report."""
    total_samples: int
    overall_quality_score: float
    task_scores: Dict[str, float]
    consistency_score: float
    sample_results: List[SampleEvaluationResult]

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        data = asdict(self)
        # Convert sample results to dicts
        data["sample_results"] = [asdict(r) for r in self.sample_results]
        return data

    def summary(self) -> str:
        """Format an executive evaluation summary."""
        lines = [
            f"Model Quality Evaluation ({self.total_samples} samples):",
            f"  Overall Score: {self.overall_quality_score:.1f}/100.0",
            f"  Deterministic Consistency: {self.consistency_score * 100:.1f}%",
            "  Task Breakdown:",
        ]
        for task, score in self.task_scores.items():
            lines.append(f"    - {task:<22}: {score:.1f}/100.0")
        return "\n".join(lines)


class Evaluator:
    """Evaluates causal language models across diverse generation tasks."""

    def __init__(self, engine: InferenceEngine, dataset: Optional[EvaluationDataset] = None):
        """Initialize Evaluator.

        Args:
            engine: Active InferenceEngine.
            dataset: EvaluationDataset instance or default.
        """
        self.engine = engine
        self.dataset = dataset or EvaluationDataset()

    def run_evaluation(
        self,
        samples: Optional[List[Dict[str, Any]]] = None,
        max_new_tokens: int = 48,
    ) -> EvaluationReport:
        """Execute evaluation across test samples and compute task-aware quality metrics.

        Args:
            samples: Optional custom list of sample dictionaries.
            max_new_tokens: Generation token limit per evaluation prompt.

        Returns:
            EvaluationReport containing per-task and aggregate quality metrics.
        """
        eval_samples = samples if samples is not None else self.dataset.get_all()
        if not eval_samples:
            logger.warning("No evaluation samples provided. Returning zero-score report.")
            return EvaluationReport(0, 0.0, {}, 1.0, [])

        sample_results: List[SampleEvaluationResult] = []
        task_scores_accum: Dict[str, List[float]] = {}

        logger.info(f"Evaluating model on {len(eval_samples)} prompt samples...")

        # 1. Evaluate task samples
        for s in eval_samples:
            sample_id = s.get("id", "unknown")
            task = s.get("task", "general")
            prompt = s.get("prompt", "")
            reference = s.get("reference", "")
            metric_type = s.get("metric", "f1_score")

            gen_result = self.engine.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Deterministic for fair evaluation
                use_cache=True,
            )
            prediction = gen_result.generated_text.strip()

            score = 0.0
            if metric_type == "exact_match":
                score = compute_exact_match(prediction, reference) * 100.0
            elif metric_type == "rouge":
                rouge_dict = compute_rouge_scores(prediction, reference)
                score = rouge_dict.get("rougeL", 0.0) * 100.0
            else:
                score = compute_token_f1(prediction, reference) * 100.0

            if task not in task_scores_accum:
                task_scores_accum[task] = []
            task_scores_accum[task].append(score)

            sample_results.append(
                SampleEvaluationResult(
                    sample_id=sample_id,
                    task=task,
                    prompt=prompt,
                    reference=reference,
                    prediction=prediction,
                    score=round(score, 2),
                    metric_name=metric_type,
                )
            )

        # 2. Check deterministic consistency across 2 consecutive runs on first sample
        consistency = 1.0
        if eval_samples:
            check_prompt = eval_samples[0]["prompt"]
            run1 = self.engine.generate(check_prompt, max_new_tokens=16, do_sample=False).generated_text
            run2 = self.engine.generate(check_prompt, max_new_tokens=16, do_sample=False).generated_text
            consistency = compute_deterministic_consistency(run1, run2)

        # 3. Aggregate metrics
        task_scores = {
            task: round(sum(scores) / len(scores), 2)
            for task, scores in task_scores_accum.items()
        }
        all_scores = [r.score for r in sample_results]
        overall_score = round(sum(all_scores) / len(all_scores), 2) if all_scores else 0.0

        report = EvaluationReport(
            total_samples=len(sample_results),
            overall_quality_score=overall_score,
            task_scores=task_scores,
            consistency_score=consistency,
            sample_results=sample_results,
        )

        logger.info(f"Evaluation complete:\n{report.summary()}")
        return report
