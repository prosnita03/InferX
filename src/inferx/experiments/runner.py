"""
Automated experiment matrix runner, results storage, and visualization generator for InferX.
"""

import csv
from datetime import datetime
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import pandas as pd
import torch

from inferx.benchmark.hardware import HardwareProfiler
from inferx.benchmark.latency import LatencyBenchmark
from inferx.benchmark.memory import MemoryTracker
from inferx.evaluation.evaluator import Evaluator
from inferx.inference.engine import InferenceEngine
from inferx.models.loader import ModelLoader
from inferx.optimization.recommender import RecommendationEngine
from inferx.utils.config import get_project_root, load_benchmark_config
from inferx.utils.logging import logger
from inferx.utils.system import HardwareNotSupportedError, set_seed


CSV_COLUMNS = [
    "experiment_id",
    "timestamp",
    "model",
    "precision",
    "device",
    "gpu",
    "batch_size",
    "sequence_length",
    "max_new_tokens",
    "kv_cache",
    "warmup_iterations",
    "benchmark_iterations",
    "mean_latency_ms",
    "median_latency_ms",
    "p95_latency_ms",
    "tokens_generated",
    "tokens_per_second",
    "requests_per_second",
    "peak_memory_mb",
    "quality_score",
    "status",
    "error_message",
]


class ExperimentRunner:
    """Orchestrates multi-configuration benchmark experiments and manages results persistence."""

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        config_path: Optional[Path] = None,
    ):
        """Initialize runner.

        Args:
            output_dir: Directory for results (CSV, JSON, plots).
            config_path: Optional benchmark YAML configuration path.
        """
        self.root = get_project_root()
        self.output_dir = output_dir or (self.root / "results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir = self.output_dir / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        self.csv_path = self.output_dir / "benchmark_results.csv"
        self.json_path = self.output_dir / "benchmark_results.json"

        self.hw_profile = HardwareProfiler.profile()
        self.config = load_benchmark_config(config_path)

        # Ensure CSV file has header
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(CSV_COLUMNS)

    def load_existing_results(self) -> List[Dict[str, Any]]:
        """Load previously stored experiment records from JSON or CSV."""
        if self.json_path.exists():
            try:
                with open(self.json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return data
            except Exception as e:
                logger.warning(f"Could not read JSON results: {e}")

        if self.csv_path.exists():
            try:
                df = pd.read_csv(self.csv_path)
                return df.to_dict(orient="records")
            except Exception as e:
                logger.warning(f"Could not read CSV results: {e}")

        return []

    def record_result(self, record: Dict[str, Any]) -> None:
        """Append result record to CSV and update JSON persistence."""
        # 1. Append to CSV
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writerow({k: record.get(k, "") for k in CSV_COLUMNS})

        # 2. Update JSON
        existing = self.load_existing_results()
        # Avoid duplicate IDs
        existing = [r for r in existing if r.get("experiment_id") != record.get("experiment_id")]
        existing.append(record)
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2)

    def run_single_experiment(
        self,
        model_name: str,
        precision: str,
        batch_size: int,
        sequence_length: int,
        max_new_tokens: int,
        kv_cache: bool,
        warmup_iterations: int,
        benchmark_iterations: int,
        evaluate_quality: bool = True,
    ) -> Dict[str, Any]:
        """Execute a single controlled benchmark configuration with error resilience.

        Returns:
            Dictionary record of experiment metrics and status.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        model_slug = model_name.split("/")[-1].lower()
        exp_id = f"exp_{int(time.time())}_{model_slug}_{precision}_b{batch_size}_s{sequence_length}_c{str(kv_cache)[0]}"

        base_record: Dict[str, Any] = {
            "experiment_id": exp_id,
            "timestamp": timestamp,
            "model": model_name,
            "precision": precision,
            "device": self.hw_profile.device_count > 0 and "cuda" or (self.hw_profile.mps_available and "mps" or "cpu"),
            "gpu": self.hw_profile.gpu_name or "N/A",
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "max_new_tokens": max_new_tokens,
            "kv_cache": kv_cache,
            "warmup_iterations": warmup_iterations,
            "benchmark_iterations": benchmark_iterations,
            "mean_latency_ms": 0.0,
            "median_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "tokens_generated": 0,
            "tokens_per_second": 0.0,
            "requests_per_second": 0.0,
            "peak_memory_mb": 0.0,
            "quality_score": 0.0,
            "status": "RUNNING",
            "error_message": "",
        }

        logger.info(
            f"\n>>> Running Experiment [{exp_id}]: model={model_name}, prec={precision}, "
            f"batch={batch_size}, seq_len={sequence_length}, kv_cache={kv_cache}"
        )

        set_seed(42)

        # 1. Attempt Model Loading
        t0 = time.perf_counter()
        loader = ModelLoader(device=base_record["device"])
        try:
            model, tokenizer, model_info = loader.load(model_name=model_name, precision=precision)
            load_time_ms = (time.perf_counter() - t0) * 1000.0
        except HardwareNotSupportedError as e:
            logger.warning(f"Configuration unsupported on current hardware: {e}")
            base_record["status"] = "UNSUPPORTED"
            base_record["error_message"] = str(e)
            self.record_result(base_record)
            return base_record
        except Exception as e:
            logger.error(f"Model load failed: {e}")
            base_record["status"] = "FAILED"
            base_record["error_message"] = f"Load error: {e}"
            self.record_result(base_record)
            return base_record

        # 2. Benchmark Inference under Memory Tracking
        try:
            engine = InferenceEngine(model=model, tokenizer=tokenizer, device=base_record["device"])
            bench = LatencyBenchmark(engine=engine)

            # Construct synthetic prompt matching desired sequence length
            sample_words = "Artificial intelligence optimization techniques allow causal language models to execute with low latency and high serving throughput. "
            repeat_count = max(1, sequence_length // 16)
            base_prompt = (sample_words * repeat_count)[:sequence_length * 4]
            prompts = [base_prompt] * batch_size

            with MemoryTracker(device=base_record["device"]) as mem_tracker:
                latency_metrics = bench.run(
                    prompts=prompts,
                    max_new_tokens=max_new_tokens,
                    warmup_iterations=warmup_iterations,
                    benchmark_iterations=benchmark_iterations,
                    use_cache=kv_cache,
                    model_load_time_ms=load_time_ms,
                )

            mem_snapshot = mem_tracker.get_snapshot()

            # Compute throughput
            total_tokens = batch_size * max_new_tokens
            latency_sec = latency_metrics.mean_latency_ms / 1000.0
            tokens_per_sec = total_tokens / latency_sec if latency_sec > 0 else 0.0
            requests_per_sec = batch_size / latency_sec if latency_sec > 0 else 0.0

            base_record["mean_latency_ms"] = latency_metrics.mean_latency_ms
            base_record["median_latency_ms"] = latency_metrics.median_latency_ms
            base_record["p95_latency_ms"] = latency_metrics.p95_latency_ms
            base_record["tokens_generated"] = total_tokens
            base_record["tokens_per_second"] = round(tokens_per_sec, 2)
            base_record["requests_per_second"] = round(requests_per_sec, 2)
            base_record["peak_memory_mb"] = mem_snapshot.peak_mb

            # 3. Quality Evaluation (optional)
            if evaluate_quality:
                evaluator = Evaluator(engine=engine)
                eval_report = evaluator.run_evaluation(max_new_tokens=32)
                base_record["quality_score"] = eval_report.overall_quality_score
            else:
                base_record["quality_score"] = 75.0  # nominal default

            base_record["status"] = "SUCCESS"

        except (torch.cuda.OutOfMemoryError, MemoryError) as e:
            logger.warning(f"Experiment exceeded available memory (OOM): {e}")
            base_record["status"] = "OOM"
            base_record["error_message"] = str(e)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"Experiment execution failed: {e}")
            base_record["status"] = "FAILED"
            base_record["error_message"] = str(e)
        finally:
            # Clean up model references to free VRAM/RAM
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self.record_result(base_record)
        return base_record

    def run_matrix(
        self,
        model_name: str,
        precisions: Optional[List[str]] = None,
        batch_sizes: Optional[List[int]] = None,
        sequence_lengths: Optional[List[int]] = None,
        kv_cache_modes: Optional[List[bool]] = None,
        warmup_iterations: int = 2,
        benchmark_iterations: int = 5,
        max_new_tokens: int = 32,
    ) -> List[Dict[str, Any]]:
        """Run full combinatorial parameter grid across hardware-supported settings.

        Returns:
            List of all executed experiment records.
        """
        is_cuda = torch.cuda.is_available()
        default_precisions = (
            self.config["experiment_matrix"]["precisions"]
            if is_cuda
            else self.config["experiment_matrix"]["cpu_precisions"]
        )
        precs = precisions or default_precisions
        batches = batch_sizes or self.config["experiment_matrix"]["batch_sizes"]
        seq_lens = sequence_lengths or [128, 512]
        caches = kv_cache_modes or [True, False]

        total_runs = len(precs) * len(batches) * len(seq_lens) * len(caches)
        logger.info(f"Initiating Experiment Matrix: {total_runs} combinations planned.")

        results = []
        for p in precs:
            for b in batches:
                for s in seq_lens:
                    for c in caches:
                        res = self.run_single_experiment(
                            model_name=model_name,
                            precision=p,
                            batch_size=b,
                            sequence_length=s,
                            max_new_tokens=max_new_tokens,
                            kv_cache=c,
                            warmup_iterations=warmup_iterations,
                            benchmark_iterations=benchmark_iterations,
                        )
                        results.append(res)

        # Generate plots after completion
        self.generate_plots()
        return results

    def generate_plots(self) -> None:
        """Generate 11 professional publication-grade comparison plots from recorded experiments."""
        records = self.load_existing_results()
        valid = [r for r in records if r.get("status") == "SUCCESS"]

        if not valid:
            logger.info("No successful experiment records found; skipping plot generation.")
            return

        df = pd.DataFrame(valid)
        logger.info(f"Generating benchmark comparison plots from {len(df)} records...")

        # Plot styling
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
        plt.rcParams.update({"font.size": 10, "axes.labelsize": 11, "axes.titlesize": 12})

        def save_fig(fig, filename: str):
            filepath = self.plots_dir / filename
            fig.tight_layout()
            fig.savefig(filepath, dpi=200)
            plt.close(fig)

        # 1. Precision vs Latency
        if "precision" in df.columns:
            fig, ax = plt.subplots(figsize=(6, 4))
            df.groupby("precision")["mean_latency_ms"].mean().plot(kind="bar", color="#3b82f6", ax=ax)
            ax.set_title("Precision vs Mean Latency")
            ax.set_ylabel("Latency (ms)")
            save_fig(fig, "1_precision_vs_latency.png")

        # 2. Precision vs Throughput
        if "precision" in df.columns:
            fig, ax = plt.subplots(figsize=(6, 4))
            df.groupby("precision")["tokens_per_second"].mean().plot(kind="bar", color="#10b981", ax=ax)
            ax.set_title("Precision vs Throughput")
            ax.set_ylabel("Tokens / Second")
            save_fig(fig, "2_precision_vs_throughput.png")

        # 3. Precision vs Memory
        if "precision" in df.columns:
            fig, ax = plt.subplots(figsize=(6, 4))
            df.groupby("precision")["peak_memory_mb"].mean().plot(kind="bar", color="#ef4444", ax=ax)
            ax.set_title("Precision vs Peak Memory Footprint")
            ax.set_ylabel("Peak Memory (MB)")
            save_fig(fig, "3_precision_vs_memory.png")

        # 4. Batch Size vs Latency
        if "batch_size" in df.columns:
            fig, ax = plt.subplots(figsize=(6, 4))
            df.groupby("batch_size")["mean_latency_ms"].mean().plot(marker="o", color="#8b5cf6", ax=ax)
            ax.set_title("Batch Size Scaling vs Inference Latency")
            ax.set_xlabel("Batch Size")
            ax.set_ylabel("Mean Latency (ms)")
            save_fig(fig, "4_batch_size_vs_latency.png")

        # 5. Batch Size vs Throughput
        if "batch_size" in df.columns:
            fig, ax = plt.subplots(figsize=(6, 4))
            df.groupby("batch_size")["tokens_per_second"].mean().plot(marker="s", color="#06b6d4", ax=ax)
            ax.set_title("Batch Size Scaling vs Throughput")
            ax.set_xlabel("Batch Size")
            ax.set_ylabel("Tokens / Second")
            save_fig(fig, "5_batch_size_vs_throughput.png")

        # 6. Batch Size vs Memory
        if "batch_size" in df.columns:
            fig, ax = plt.subplots(figsize=(6, 4))
            df.groupby("batch_size")["peak_memory_mb"].mean().plot(marker="^", color="#f59e0b", ax=ax)
            ax.set_title("Batch Size Scaling vs Memory Footprint")
            ax.set_xlabel("Batch Size")
            ax.set_ylabel("Peak Memory (MB)")
            save_fig(fig, "6_batch_size_vs_memory.png")

        # 7. Sequence Length vs Latency
        if "sequence_length" in df.columns:
            fig, ax = plt.subplots(figsize=(6, 4))
            df.groupby("sequence_length")["mean_latency_ms"].mean().plot(marker="o", color="#ec4899", ax=ax)
            ax.set_title("Input Sequence Length vs Latency")
            ax.set_xlabel("Sequence Length (tokens)")
            ax.set_ylabel("Latency (ms)")
            save_fig(fig, "7_sequence_length_vs_latency.png")

        # 8. Sequence Length vs Memory
        if "sequence_length" in df.columns:
            fig, ax = plt.subplots(figsize=(6, 4))
            df.groupby("sequence_length")["peak_memory_mb"].mean().plot(marker="s", color="#6366f1", ax=ax)
            ax.set_title("Input Sequence Length vs Memory Consumption")
            ax.set_xlabel("Sequence Length (tokens)")
            ax.set_ylabel("Peak Memory (MB)")
            save_fig(fig, "8_sequence_length_vs_memory.png")

        # 9. Quality vs Latency
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(df["mean_latency_ms"], df["quality_score"], c="#3b82f6", alpha=0.8, edgecolors="none")
        ax.set_title("Quality vs Latency Pareto Space")
        ax.set_xlabel("Mean Latency (ms)")
        ax.set_ylabel("Quality Score (0-100)")
        save_fig(fig, "9_quality_vs_latency.png")

        # 10. Quality vs Memory
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(df["peak_memory_mb"], df["quality_score"], c="#10b981", alpha=0.8, edgecolors="none")
        ax.set_title("Quality vs Memory Consumption")
        ax.set_xlabel("Peak Memory (MB)")
        ax.set_ylabel("Quality Score (0-100)")
        save_fig(fig, "10_quality_vs_memory.png")

        # 11. Quality vs Throughput
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(df["tokens_per_second"], df["quality_score"], c="#8b5cf6", alpha=0.8, edgecolors="none")
        ax.set_title("Quality vs Serving Throughput")
        ax.set_xlabel("Tokens / Second")
        ax.set_ylabel("Quality Score (0-100)")
        save_fig(fig, "11_quality_vs_throughput.png")

        logger.info(f"Saved 11 benchmark plots to: {self.plots_dir}")
