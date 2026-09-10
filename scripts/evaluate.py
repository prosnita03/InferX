#!/usr/bin/env python3
"""
InferX Model Quality Evaluation CLI.

Evaluates causal language models across multi-task benchmarks (QA, factual recall,
reasoning, and summarization) with task-aware metrics (Exact Match, F1, ROUGE).

Usage:
    python scripts/evaluate.py --demo
    python scripts/evaluate.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
"""

import argparse
import sys
from pathlib import Path
import torch

# Ensure repository root is on Python path and prevent shadowing stdlib modules (e.g. profile.py)
scripts_dir = str(Path(__file__).resolve().parent)
while scripts_dir in sys.path:
    sys.path.remove(scripts_dir)

src_dir = str(Path(__file__).resolve().parent.parent / "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from inferx.evaluation.dataset import EvaluationDataset
from inferx.evaluation.evaluator import Evaluator
from inferx.inference.engine import InferenceEngine
from inferx.models.loader import ModelLoader
from inferx.utils.config import load_models_config
from inferx.utils.logging import logger
from inferx.utils.system import set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run multi-task NLP quality evaluation on causal language models using InferX.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Hugging Face model repository ID.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default=None,
        choices=["fp32", "fp16", "bf16", "int8", "int4"],
        help="Precision format.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to custom JSON evaluation dataset file.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
        help="Maximum generation tokens per evaluation prompt.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run rapid demonstration on sample subset.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(42)

    cfg = load_models_config()
    if args.demo:
        model_name = args.model or cfg.get("demo_model", "hf-internal-testing/tiny-random-gpt2")
        precision = args.precision or "fp32"
        logger.info(f"DEMO MODE: Evaluating on demo subset with model '{model_name}'.")
    else:
        model_name = args.model or cfg.get("default_model", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        is_cuda = torch.cuda.is_available()
        precision = args.precision or ("fp16" if is_cuda else "fp32")

    print("==================================================")
    print("InferX — Task-Aware Quality Evaluation Suite")
    print("==================================================")
    print(f"Model:     {model_name}")
    print(f"Precision: {precision.upper()}")
    print("==================================================\n")

    loader = ModelLoader()
    model, tokenizer, model_info = loader.load(model_name=model_name, precision=precision)
    engine = InferenceEngine(model=model, tokenizer=tokenizer, device=loader.device)

    dataset_path = Path(args.dataset) if args.dataset else None
    eval_dataset = EvaluationDataset(dataset_path=dataset_path)

    samples = eval_dataset.get_demo_samples(3) if args.demo else eval_dataset.get_all()

    evaluator = Evaluator(engine=engine, dataset=eval_dataset)
    report = evaluator.run_evaluation(samples=samples, max_new_tokens=args.max_new_tokens)

    print("\n" + "=" * 50)
    print(report.summary())
    print("=" * 50)

    print("\nDetailed Sample Predictions:")
    for s in report.sample_results:
        print(f"\n[ID: {s.sample_id} | Task: {s.task}]")
        print(f"  Prompt:    {s.prompt.strip()[:60]}...")
        print(f"  Target:    {s.reference}")
        print(f"  Generated: {s.prediction}")
        print(f"  Score:     {s.score:.1f}% ({s.metric_name})")


if __name__ == "__main__":
    main()
