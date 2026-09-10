"""
PyTorch profiler integration for low-level operator and kernel inspection.
"""

from pathlib import Path
from typing import Optional
import torch

from inferx.inference.engine import InferenceEngine
from inferx.utils.logging import logger


class PyTorchProfiler:
    """Wrapper around torch.profiler for operator-level execution and memory tracing."""

    def __init__(self, engine: InferenceEngine, output_dir: Optional[Path] = None):
        """Initialize profiler.

        Args:
            engine: Active InferenceEngine.
            output_dir: Directory to save trace JSON files.
        """
        self.engine = engine
        self.device = engine.device
        self.output_dir = output_dir or Path("results/traces")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def profile_inference(
        self,
        prompt: str = "Explain transformer self-attention mechanisms in deep neural networks.",
        max_new_tokens: int = 16,
        record_shapes: bool = True,
        profile_memory: bool = True,
        with_stack: bool = False,
    ) -> str:
        """Run profiled inference and return textual operator table report.

        Args:
            prompt: Text prompt string.
            max_new_tokens: Generation budget.
            record_shapes: Record tensor dimensions.
            profile_memory: Track tensor memory allocations.
            with_stack: Record Python source code stack.

        Returns:
            Formatted string table of top operators by execution time.
        """
        is_cuda = "cuda" in self.device and torch.cuda.is_available()

        activities = [torch.profiler.ProfilerActivity.CPU]
        if is_cuda:
            activities.append(torch.profiler.ProfilerActivity.CUDA)
            logger.info("Enabling CPU + CUDA activity profiling.")
        else:
            logger.info("CUDA not active; profiling CPU operations.")

        # Warmup pass
        self.engine.warmup(iterations=1, max_new_tokens=4)

        with torch.profiler.profile(
            activities=activities,
            record_shapes=record_shapes,
            profile_memory=profile_memory,
            with_stack=with_stack,
        ) as prof:
            _ = self.engine.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )

        # Generate summary table
        sort_by = "cuda_time_total" if is_cuda else "cpu_time_total"
        summary_table = prof.key_averages().table(
            sort_by=sort_by,
            row_limit=20,
            max_name_column_width=45,
        )

        # Export trace
        trace_file = self.output_dir / f"trace_{'cuda' if is_cuda else 'cpu'}.json"
        try:
            prof.export_chrome_trace(str(trace_file))
            logger.info(f"Chrome trace exported to: {trace_file}")
        except Exception as e:
            logger.warning(f"Could not export Chrome trace: {e}")

        return summary_table
