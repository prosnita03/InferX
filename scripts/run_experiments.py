#!/usr/bin/env python3
"""
InferX Automated Experiment Matrix Runner.

Executes combinatorial parameter sweeps across precisions, batch sizes,
sequence lengths, and KV-cache settings, persisting results and outputting
Pareto optimization recommendations.

Usage:
    python scripts/run_experiments.py --demo
    python scripts/run_experiments.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
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

from inferx.experiments.runner import ExperimentRunner
from inferx.optimization.recommender import RecommendationEngine
from inferx.utils.config import load_benchmark_config, load_models_config
from inferx.utils.logging import logger
from inferx.utils.system import set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Execute automated combinatorial LLM inference experiments using InferX.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Hugging Face model repository ID.",
    )
    parser.add_argument(
        "--precisions",
        nargs="+",
        default=None,
        help="Precisions to benchmark (e.g. fp16 int8 int4 fp32).",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=None,
        help="Batch sizes to evaluate (e.g. 1 2 4 8).",
    )
    parser.add_argument(
        "--sequence-lengths",
        type=int,
        nargs="+",
        default=None,
        help="Sequence lengths in tokens (e.g. 128 512).",
    )
    parser.add_argument(
        "--objective",
        type=str,
        default="balanced",
        choices=["balanced", "low_latency", "high_throughput", "low_memory", "high_quality"],
        help="Optimization objective for recommendation report.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run compact demonstration grid on micro-scale model for rapid validation.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(42)

    models_cfg = load_models_config()
    bench_cfg = load_benchmark_config()

    if args.demo:
        model_name = args.model or models_cfg.get("demo_model", "hf-internal-testing/tiny-random-gpt2")
        precisions = args.precisions or ["fp32"]
        batch_sizes = args.batch_sizes or [1, 2]
        sequence_lengths = args.sequence_lengths or [32, 64]
        kv_cache_modes = [True, False]
        warmup = 1
        iterations = 2
        max_tokens = 16
        logger.info(f"DEMO EXPERIMENT MATRIX: Running verification suite on '{model_name}'.")
    else:
        model_name = args.model or models_cfg.get("default_model", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        is_cuda = torch.cuda.is_available()
        default_precs = (
            bench_cfg["experiment_matrix"]["precisions"]
            if is_cuda
            else bench_cfg["experiment_matrix"]["cpu_precisions"]
        )
        precisions = args.precisions or default_precs
        batch_sizes = args.batch_sizes or bench_cfg["experiment_matrix"]["batch_sizes"]
        sequence_lengths = args.sequence_lengths or [128, 512]
        kv_cache_modes = [True, False]
        warmup = bench_cfg["defaults"]["warmup_iterations"]
        iterations = bench_cfg["defaults"]["benchmark_iterations"]
        max_tokens = bench_cfg["defaults"]["max_new_tokens"]

    print("==================================================")
    print("InferX — Experiment Matrix Suite")
    print("==================================================")
    print(f"Target Model:     {model_name}")
    print(f"Precisions:       {precisions}")
    print(f"Batch Sizes:      {batch_sizes}")
    print(f"Sequence Lengths: {sequence_lengths}")
    print(f"KV Cache Modes:   {kv_cache_modes}")
    print(f"Iterations:       {iterations} (Warmup: {warmup})")
    print("==================================================\n")

    runner = ExperimentRunner()
    results = runner.run_matrix(
        model_name=model_name,
        precisions=precisions,
        batch_sizes=batch_sizes,
        sequence_lengths=sequence_lengths,
        kv_cache_modes=kv_cache_modes,
        warmup_iterations=warmup,
        benchmark_iterations=iterations,
        max_new_tokens=max_tokens,
    )

    print("\n" + "=" * 50)
    print(f"Experiment Matrix Complete: {len(results)} configurations processed.")
    print(f"Results persisted to:\n  - {runner.csv_path}\n  - {runner.json_path}")
    print(f"Visualizations saved to:\n  - {runner.plots_dir}")
    print("=" * 50 + "\n")

    # Generate Pareto Optimization Recommendation
    rec_engine = RecommendationEngine(objective=args.objective)
    report = rec_engine.recommend(results)
    if report:
        print(report.summary())


if __name__ == "__main__":
    main()
