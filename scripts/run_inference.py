#!/usr/bin/env python3
"""
InferX Inference Runner CLI.

Executes text-generation inference with precision control and detailed latency tracking.

Usage:
    python scripts/run_inference.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --prompt "Explain transformers simply."
    python scripts/run_inference.py --demo
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

from inferx.inference.engine import InferenceEngine
from inferx.models.loader import ModelLoader
from inferx.utils.config import load_models_config
from inferx.utils.logging import logger
from inferx.utils.system import format_latency, format_throughput, set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run text-generation inference with latency profiling using InferX.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Hugging Face causal language model identifier or local path.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Explain transformers simply.",
        help="Input text prompt for inference generation.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default=None,
        choices=["fp32", "fp16", "bf16", "int8", "int4"],
        help="Precision configuration (defaults to fp16 on CUDA, fp32 on CPU).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (when do-sample is enabled).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling probability.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling threshold.",
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Enable stochastic sampling instead of greedy deterministic decoding.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable past Key/Value state caching (use_cache=False).",
    )
    parser.add_argument(
        "--measure-tokens",
        action="store_true",
        help="Profile fine-grained step-by-step TTFT and inter-token latencies.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Execute rapid demonstration using a micro-scale synthetic model.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(42)

    cfg = load_models_config()
    model_name = args.model
    if args.demo:
        model_name = cfg.get("demo_model", "hf-internal-testing/tiny-random-gpt2")
        logger.info(f"Demo Mode Active: Using lightweight verification model '{model_name}'.")
    elif not model_name:
        model_name = cfg.get("default_model", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    # Determine default precision based on hardware
    is_cuda = torch.cuda.is_available()
    precision = args.precision or ("fp16" if is_cuda else "fp32")

    use_cache = not args.no_cache

    print("==================================================")
    print("InferX — LLM Inference Engine")
    print("==================================================")
    print(f"Model:           {model_name}")
    print(f"Precision:       {precision.upper()}")
    print(f"Max New Tokens:  {args.max_new_tokens}")
    print(f"KV Cache:        {'Enabled' if use_cache else 'Disabled'}")
    print(f"Deterministic:   {not args.do_sample}")
    print("==================================================\n")

    # 1. Load model and tokenizer
    loader = ModelLoader()
    model, tokenizer, model_info = loader.load(model_name=model_name, precision=precision)

    # 2. Initialize Inference Engine
    engine = InferenceEngine(model=model, tokenizer=tokenizer, device=loader.device)

    # 3. Warm-up
    engine.warmup(iterations=1, max_new_tokens=8)

    # 4. Execute generation
    print(f"\n[Prompt]:\n{args.prompt}\n")

    if args.measure_tokens:
        print("Measuring token-by-token generation latencies...")
        res = engine.measure_token_latencies(
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            do_sample=args.do_sample,
            use_cache=use_cache,
        )
        print(f"\n[Generated Output]:\n{res.generated_text}\n")
        print("--------------------------------------------------")
        print("Detailed Timing Breakdown:")
        print(f"  Time-To-First-Token (TTFT): {format_latency(res.time_to_first_token_ms)}")
        print(f"  Mean Inter-Token Latency:   {format_latency(res.mean_inter_token_latency_ms)}")
        print(f"  Total Decode Latency:       {format_latency(res.decode_latency_ms)}")
        print(f"  Total End-to-End Latency:   {format_latency(res.total_latency_ms)}")
        print(f"  Throughput:                 {format_throughput(res.tokens_per_second)}")
        print(f"  Tokens Generated:           {res.generated_tokens} tokens")
        print("--------------------------------------------------")
    else:
        res = engine.generate(
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            do_sample=args.do_sample,
            use_cache=use_cache,
        )
        print(f"\n[Generated Output]:\n{res.generated_text}\n")
        print("--------------------------------------------------")
        print(f"Total Latency:    {format_latency(res.total_latency_ms)}")
        print(f"Throughput:       {format_throughput(res.tokens_per_second)}")
        print(f"Tokens Generated: {res.generated_tokens} tokens")
        print("--------------------------------------------------")


if __name__ == "__main__":
    main()
