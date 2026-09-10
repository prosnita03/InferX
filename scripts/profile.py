#!/usr/bin/env python3
"""
InferX PyTorch Operator Profiler CLI.

Profiles low-level neural network operators, CUDA kernels, and memory allocations
using torch.profiler.

Usage:
    python scripts/profile.py --demo
    python scripts/profile.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
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

from inferx.benchmark.profiler import PyTorchProfiler
from inferx.inference.engine import InferenceEngine
from inferx.models.loader import ModelLoader
from inferx.utils.config import load_models_config
from inferx.utils.logging import logger
from inferx.utils.system import set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Profile PyTorch operator execution time and memory activity using InferX.",
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
        help="Model precision.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Explain transformer self-attention mechanisms in deep neural networks.",
        help="Input prompt to profile during forward generation.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=16,
        help="Generation step count for profiling pass.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Profile a micro-scale model for rapid demonstration.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(42)

    cfg = load_models_config()
    if args.demo:
        model_name = args.model or cfg.get("demo_model", "hf-internal-testing/tiny-random-gpt2")
        precision = args.precision or "fp32"
    else:
        model_name = args.model or cfg.get("default_model", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        is_cuda = torch.cuda.is_available()
        precision = args.precision or ("fp16" if is_cuda else "fp32")

    print("==================================================")
    print("InferX — PyTorch Low-Level Operator Profiler")
    print("==================================================")
    print(f"Model:     {model_name}")
    print(f"Precision: {precision.upper()}")
    print(f"CUDA:      {'Available' if torch.cuda.is_available() else 'Not available (profiling CPU ops)'}")
    print("==================================================\n")

    loader = ModelLoader()
    model, tokenizer, model_info = loader.load(model_name=model_name, precision=precision)
    engine = InferenceEngine(model=model, tokenizer=tokenizer, device=loader.device)

    profiler = PyTorchProfiler(engine=engine)
    print("Running operator execution profiling...")
    report_table = profiler.profile_inference(
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        record_shapes=True,
        profile_memory=True,
    )

    print("\n" + "=" * 80)
    print("TOP OPERATOR EXECUTION TIME & MEMORY PROFILE:")
    print("=" * 80)
    print(report_table)
    print("=" * 80)
    print(f"Chrome trace exported to: {profiler.output_dir}\n")


if __name__ == "__main__":
    main()
