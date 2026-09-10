"""
InferX — Enterprise LLM Inference Optimization & Benchmarking Platform.

A high-performance, minimalist dark-themed interface for analyzing latency distributions,
serving throughput, memory hierarchies, quantization trade-offs, and Pareto efficiency frontiers.
"""

import os
import sys
import time
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch

# Ensure src/ is on Python search path and prevent shadowing stdlib modules
scripts_dir = str(Path(__file__).resolve().parent)
while scripts_dir in sys.path:
    sys.path.remove(scripts_dir)

src_dir = str(Path(__file__).resolve().parent.parent / "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from inferx.benchmark.hardware import HardwareProfiler
from inferx.benchmark.latency import LatencyBenchmark
from inferx.benchmark.memory import MemoryTracker
from inferx.experiments.runner import ExperimentRunner
from inferx.inference.engine import InferenceEngine
from inferx.models.loader import ModelLoader
from inferx.optimization.recommender import RecommendationEngine
from inferx.optimization.scoring import OBJECTIVE_PRESETS
from inferx.utils.config import load_benchmark_config, load_models_config
from inferx.utils.system import format_latency, format_throughput, set_seed

# Page Configuration
st.set_page_config(
    page_title="InferX | LLM Inference Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Minimalist Obsidian Design System with Glassmorphism
st.markdown(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">

    <style>
    /* Global System Variables */
    :root {
        --bg-surface: #06090e;
        --card-bg: rgba(13, 18, 30, 0.7);
        --card-border: rgba(255, 255, 255, 0.08);
        --card-hover-border: rgba(99, 102, 241, 0.4);
        --accent-primary: #6366f1;
        --accent-teal: #14b8a6;
        --accent-emerald: #10b981;
        --accent-violet: #8b5cf6;
        --text-headline: #f8fafc;
        --text-body: #cbd5e1;
        --text-subtle: #64748b;
        --code-font: 'JetBrains Mono', monospace;
    }

    .stApp {
        background: radial-gradient(circle at 12% 14%, rgba(99, 102, 241, 0.09) 0%, transparent 40%),
                    radial-gradient(circle at 88% 18%, rgba(139, 92, 246, 0.08) 0%, transparent 45%),
                    radial-gradient(circle at 50% 88%, rgba(20, 184, 166, 0.06) 0%, transparent 40%),
                    #070a11;
        color: var(--text-headline);
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Sidebar Navigation */
    [data-testid="stSidebar"] {
        background-color: rgba(9, 13, 23, 0.88);
        border-right: 1px solid rgba(255, 255, 255, 0.06);
        backdrop-filter: blur(24px);
    }

    /* Brand Logo Strip */
    .brand-strip {
        display: flex;
        align-items: center;
        gap: 12px;
        padding-bottom: 20px;
        margin-bottom: 20px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.07);
    }

    .brand-mark {
        width: 34px;
        height: 34px;
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 800;
        font-size: 1rem;
        color: #ffffff;
        letter-spacing: -0.05em;
        box-shadow: 0 4px 14px rgba(79, 70, 229, 0.35);
    }

    .brand-title {
        font-size: 1.15rem;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: -0.02em;
        line-height: 1.1;
    }

    .brand-subtitle {
        font-size: 0.7rem;
        color: #818cf8;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    /* Hero Banner */
    .hero-container {
        background: linear-gradient(135deg, rgba(18, 25, 42, 0.7) 0%, rgba(11, 16, 28, 0.8) 100%);
        border: 1px solid rgba(255, 255, 255, 0.09);
        border-radius: 18px;
        padding: 30px 34px;
        margin-bottom: 26px;
        backdrop-filter: blur(16px);
        position: relative;
        box-shadow: 0 16px 36px -12px rgba(0, 0, 0, 0.5);
    }

    .hero-container::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; height: 1px;
        background: linear-gradient(90deg, transparent, rgba(99, 102, 241, 0.5), rgba(139, 92, 246, 0.5), transparent);
    }

    .hero-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(99, 102, 241, 0.12);
        border: 1px solid rgba(99, 102, 241, 0.3);
        color: #a5b4fc;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.09em;
        text-transform: uppercase;
        padding: 4px 10px;
        border-radius: 9999px;
        margin-bottom: 12px;
    }

    .hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        background: linear-gradient(90deg, #ffffff 0%, #e2e8f0 60%, #a5b4fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
        line-height: 1.2;
    }

    .hero-desc {
        color: #94a3b8;
        font-size: 1.02rem;
        max-width: 820px;
        line-height: 1.55;
        font-weight: 400;
    }

    /* Minimalist Stat Cards */
    .stat-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 14px;
        margin-bottom: 26px;
    }

    .stat-card {
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        border-radius: 14px;
        padding: 18px 20px;
        backdrop-filter: blur(14px);
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
    }

    .stat-card:hover {
        transform: translateY(-3px);
        border-color: var(--card-hover-border);
        box-shadow: 0 10px 24px -6px rgba(99, 102, 241, 0.2);
    }

    .stat-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 6px;
    }

    .stat-tag {
        font-size: 0.72rem;
        color: #818cf8;
        font-family: var(--code-font);
        font-weight: 600;
        letter-spacing: 0.04em;
        background: rgba(99, 102, 241, 0.1);
        padding: 2px 7px;
        border-radius: 5px;
    }

    .stat-label {
        font-size: 0.78rem;
        color: #94a3b8;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }

    .stat-value {
        font-size: 1.85rem;
        font-weight: 700;
        font-family: var(--code-font);
        color: #ffffff;
        margin-bottom: 2px;
        letter-spacing: -0.02em;
    }

    .stat-footnote {
        font-size: 0.8rem;
        color: var(--text-subtle);
    }

    /* Status Dot & Live Indicator */
    .live-status {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.25);
        color: #34d399;
        padding: 5px 12px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        font-family: var(--code-font);
    }

    .pulse-dot {
        width: 7px;
        height: 7px;
        background-color: #10b981;
        border-radius: 50%;
        box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
        animation: pulse-ring 1.8s infinite cubic-bezier(0.66, 0, 0, 1);
    }

    @keyframes pulse-ring {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 7px rgba(16, 185, 129, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #4f46e5 0%, #6366f1 100%) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 10px !important;
        padding: 9px 22px !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
        letter-spacing: 0.01em !important;
        box-shadow: 0 4px 16px rgba(79, 70, 229, 0.3) !important;
        transition: all 0.2s ease !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 22px rgba(79, 70, 229, 0.45) !important;
        border-color: rgba(255, 255, 255, 0.25) !important;
    }

    /* Recommendation Showcase */
    .rec-box {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.5) 0%, rgba(15, 23, 42, 0.7) 100%);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 22px;
    }

    .rec-title {
        font-size: 1.3rem;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 12px;
        letter-spacing: -0.01em;
    }

    .meta-tag {
        display: inline-block;
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 4px 10px;
        border-radius: 6px;
        font-family: var(--code-font);
        font-size: 0.82rem;
        font-weight: 600;
        color: #e2e8f0;
        margin-right: 6px;
        margin-bottom: 6px;
    }

    .meta-tag-highlight {
        background: rgba(99, 102, 241, 0.22);
        border-color: rgba(99, 102, 241, 0.5);
        color: #ffffff;
    }

    /* Tables & Inputs */
    [data-testid="stDataFrame"] {
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        overflow: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_hardware_info():
    """Retrieve host accelerator and CPU characteristics."""
    return HardwareProfiler.profile()


@st.cache_data
def get_model_options():
    """Load model registry configuration."""
    cfg = load_models_config()
    return cfg.get("models", {}), cfg.get("default_model", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")


def load_dataset_results():
    """Load recorded benchmark results from disk."""
    runner = ExperimentRunner()
    return runner.load_existing_results()


def apply_premium_chart_theme(fig: go.Figure) -> go.Figure:
    """Apply clean minimalist dark styling to Plotly figures."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13, 18, 30, 0.4)",
        font=dict(family="Plus Jakarta Sans, sans-serif", color="#94a3b8", size=11),
        title_font=dict(family="Plus Jakarta Sans, sans-serif", color="#f8fafc", size=14),
        margin=dict(l=40, r=20, t=45, b=40),
        xaxis=dict(
            gridcolor="rgba(255, 255, 255, 0.04)",
            linecolor="rgba(255, 255, 255, 0.08)",
            tickfont=dict(color="#64748b", family="JetBrains Mono, monospace"),
        ),
        yaxis=dict(
            gridcolor="rgba(255, 255, 255, 0.04)",
            linecolor="rgba(255, 255, 255, 0.08)",
            tickfont=dict(color="#64748b", family="JetBrains Mono, monospace"),
        ),
        legend=dict(
            bgcolor="rgba(13, 18, 30, 0.7)",
            bordercolor="rgba(255, 255, 255, 0.08)",
            borderwidth=1,
            font=dict(color="#cbd5e1"),
        ),
        hoverlabel=dict(
            bgcolor="rgba(13, 18, 30, 0.95)",
            bordercolor="rgba(99, 102, 241, 0.4)",
            font=dict(family="JetBrains Mono, monospace", color="#ffffff", size=12),
        ),
    )
    return fig


def main():
    hw = get_hardware_info()
    models_dict, default_model_name = get_model_options()

    # Sidebar Navigation & System Status
    with st.sidebar:
        st.markdown(
            """
            <div class="brand-strip">
                <div class="brand-mark">IX</div>
                <div>
                    <div class="brand-title">InferX</div>
                    <div class="brand-subtitle">Inference Optimization</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        nav_mode = st.radio(
            "Navigation",
            [
                "Overview",
                "Live Profiler",
                "Quantization Analysis",
                "Concurrency & Batching",
                "Context Scaling",
                "Pareto Frontier",
                "Recommendations",
                "Audit Log",
            ],
            index=0,
        )

        st.markdown("<hr style='border: none; border-top: 1px solid rgba(255,255,255,0.06); margin: 20px 0;'>", unsafe_allow_html=True)
        st.markdown("<div style='font-size: 0.72rem; color: #64748b; font-weight: 700; text-transform: uppercase; letter-spacing: 0.07em; margin-bottom: 10px;'>Runtime Accelerator</div>", unsafe_allow_html=True)

        if hw.cuda_available:
            st.markdown('<div class="live-status"><div class="pulse-dot"></div> NVIDIA CUDA ACTIVE</div>', unsafe_allow_html=True)
            st.markdown(f"<div style='font-size: 0.83rem; color: #cbd5e1; margin-top: 8px;'><b>GPU:</b> {hw.gpu_name}</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size: 0.8rem; color: #94a3b8;'><b>VRAM:</b> {hw.gpu_total_memory_gb:.1f} GB Dedicated</div>", unsafe_allow_html=True)
        elif hw.mps_available:
            st.markdown('<div class="live-status"><div class="pulse-dot"></div> APPLE SILICON (MPS)</div>', unsafe_allow_html=True)
            st.markdown(f"<div style='font-size: 0.83rem; color: #cbd5e1; margin-top: 8px;'><b>Device:</b> {hw.cpu_model}</div>", unsafe_allow_html=True)
        else:
            st.markdown('<div class="live-status" style="background: rgba(245, 158, 11, 0.1); color: #fbbf24; border-color: rgba(245, 158, 11, 0.25);"><div class="pulse-dot" style="background-color: #f59e0b;"></div> CPU BACKEND</div>', unsafe_allow_html=True)
            st.markdown(f"<div style='font-size: 0.83rem; color: #cbd5e1; margin-top: 8px;'><b>CPU:</b> {hw.cpu_model} ({hw.cpu_cores_logical} cores)</div>", unsafe_allow_html=True)

        st.markdown(f"<div style='font-size: 0.78rem; color: #64748b; margin-top: 4px;'>PyTorch {hw.pytorch_version} &bull; RAM {hw.system_ram_gb:.1f} GB</div>", unsafe_allow_html=True)

    results = load_dataset_results()
    valid_results = [r for r in results if r.get("status") == "SUCCESS"]

    # =========================================================================
    # 1. OVERVIEW
    # =========================================================================
    if nav_mode == "Overview":
        st.markdown(
            """
            <div class="hero-container">
                <div class="hero-pill">INFERENCE BENCHMARKING ENGINE</div>
                <div class="hero-title">InferX — LLM Inference Optimization Platform</div>
                <div class="hero-desc">
                    Systematic empirical benchmarking, CUDA stream synchronization, memory footprint profiling, 
                    and multi-objective Pareto optimization for generative causal language models.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        mean_lat = f"{valid_results[-1]['mean_latency_ms']:.1f} ms" if valid_results else "N/A"
        mean_thr = f"{valid_results[-1]['tokens_per_second']:.1f} tok/s" if valid_results else "N/A"
        peak_mem = f"{valid_results[-1]['peak_memory_mb']:.1f} MB" if valid_results else "N/A"

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-header">
                        <span class="stat-label">Execution Backend</span>
                        <span class="stat-tag">BACKEND</span>
                    </div>
                    <div class="stat-value">{'CUDA' if hw.cuda_available else ('MPS' if hw.mps_available else 'CPU')}</div>
                    <div class="stat-footnote">{'Tensor Core Acceleration' if hw.cuda_available else 'Apple Metal Shaders'}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-header">
                        <span class="stat-label">Benchmark Runs</span>
                        <span class="stat-tag">METRICS</span>
                    </div>
                    <div class="stat-value">{len(valid_results)}</div>
                    <div class="stat-footnote">Synchronized empirical passes</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-header">
                        <span class="stat-label">Latest Throughput</span>
                        <span class="stat-tag">THROUGHPUT</span>
                    </div>
                    <div class="stat-value">{mean_thr}</div>
                    <div class="stat-footnote">Batched token generation rate</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c4:
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-header">
                        <span class="stat-label">Peak Memory</span>
                        <span class="stat-tag">FOOTPRINT</span>
                    </div>
                    <div class="stat-value">{peak_mem}</div>
                    <div class="stat-footnote">VRAM / RSS allocation</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("### Execution Regimes")
        st.markdown(
            """
            <div style="background: rgba(13, 18, 30, 0.45); border: 1px solid rgba(255, 255, 255, 0.07); border-radius: 12px; padding: 18px 22px; margin-bottom: 22px;">
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 14px;">
                    <div style="background: rgba(20, 27, 45, 0.4); border: 1px solid rgba(99, 102, 241, 0.18); border-radius: 10px; padding: 14px 16px;">
                        <div style="color: #818cf8; font-weight: 700; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em;">1. Prefill Phase (Prompt Processing)</div>
                        <p style="color: #94a3b8; font-size: 0.83rem; margin-top: 4px; margin-bottom: 0; line-height: 1.5;">Compute-bound matrix multiplication over the entire prompt context. Governs <b>Time-To-First-Token (TTFT)</b>.</p>
                    </div>
                    <div style="background: rgba(20, 27, 45, 0.4); border: 1px solid rgba(16, 185, 129, 0.18); border-radius: 10px; padding: 14px 16px;">
                        <div style="color: #34d399; font-weight: 700; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em;">2. Decode Phase (Autoregressive Generation)</div>
                        <p style="color: #94a3b8; font-size: 0.83rem; margin-top: 4px; margin-bottom: 0; line-height: 1.5;">Memory-bandwidth bound vector-matrix multiplications. Governs <b>Inter-Token Latency (ITL)</b> and KV-cache expansion.</p>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if valid_results:
            df = pd.DataFrame(valid_results)
            st.markdown("### Empirical Benchmark Snapshot")
            cols_to_show = ["experiment_id", "model", "precision", "batch_size", "sequence_length", "mean_latency_ms", "tokens_per_second", "peak_memory_mb", "status"]
            st.dataframe(df[cols_to_show].tail(8), use_container_width=True)
        else:
            st.info("No recorded benchmarks yet. Navigate to Live Profiler or run `python scripts/benchmark.py --demo`.")

    # =========================================================================
    # 2. LIVE PROFILER
    # =========================================================================
    elif nav_mode == "Live Profiler":
        st.markdown("## Live Latency & Memory Profiler")
        st.caption("Synchronized device execution measuring TTFT, inter-token latencies, and memory footprints.")

        model_choices = list(models_dict.keys())
        default_idx = model_choices.index(default_model_name) if default_model_name in model_choices else 0

        with st.container():
            c1, c2, c3 = st.columns(3)
            with c1:
                selected_model = st.selectbox("Target Architecture", model_choices, index=default_idx)
                precision = st.selectbox("Precision Format", ["fp32", "fp16", "int8", "int4"], index=1 if hw.cuda_available else 0)
            with c2:
                batch_size = st.slider("Serving Concurrency (Batch Size)", min_value=1, max_value=8, value=1)
                max_new_tokens = st.slider("Generation Token Budget", min_value=8, max_value=128, value=32, step=8)
            with c3:
                sequence_length = st.select_slider("Context Length (tokens)", options=[32, 64, 128, 256, 512], value=64)
                kv_cache = st.checkbox("Enable Past KV-Cache", value=True)
                num_iters = st.number_input("Benchmark Iterations", min_value=1, max_value=20, value=3)

        if st.button("Run Benchmark Pass", type="primary"):
            with st.spinner("Executing synchronized benchmark pass on hardware..."):
                runner = ExperimentRunner()
                res = runner.run_single_experiment(
                    model_name=selected_model,
                    precision=precision,
                    batch_size=batch_size,
                    sequence_length=sequence_length,
                    max_new_tokens=max_new_tokens,
                    kv_cache=kv_cache,
                    warmup_iterations=1,
                    benchmark_iterations=num_iters,
                    evaluate_quality=False,
                )

                if res.get("status") == "SUCCESS":
                    st.success(f"Benchmark run completed. ID: `{res['experiment_id']}`")
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Mean Latency", format_latency(res["mean_latency_ms"]))
                    m2.metric("P95 Tail Latency", format_latency(res["p95_latency_ms"]))
                    m3.metric("Throughput", format_throughput(res["tokens_per_second"]))
                    m4.metric("Peak Memory", f"{res['peak_memory_mb']:.1f} MB")
                else:
                    st.error(f"Configuration flagged: {res.get('status')} — {res.get('error_message')}")

    # =========================================================================
    # 3. QUANTIZATION DYNAMICS
    # =========================================================================
    elif nav_mode == "Quantization Analysis":
        st.markdown("## Quantization Dynamics: Precision vs Memory & Speed")
        st.caption("Empirical memory compaction and throughput trade-offs across FP16, INT8, and INT4.")

        if not valid_results:
            st.info("Run `python scripts/run_experiments.py --demo` to populate data.")
            return

        df = pd.DataFrame(valid_results)
        c1, c2 = st.columns(2)
        with c1:
            fig_mem = px.bar(
                df, x="precision", y="peak_memory_mb", color="precision",
                title="Peak Memory Footprint across Precisions (MB)",
                color_discrete_sequence=["#6366f1", "#10b981", "#f59e0b", "#06b6d4"],
            )
            st.plotly_chart(apply_premium_chart_theme(fig_mem), use_container_width=True)

        with c2:
            fig_lat = px.bar(
                df, x="precision", y="mean_latency_ms", color="precision",
                title="Steady-State Latency across Precisions (ms)",
                color_discrete_sequence=["#8b5cf6", "#ec4899", "#3b82f6", "#14b8a6"],
            )
            st.plotly_chart(apply_premium_chart_theme(fig_lat), use_container_width=True)

    # =========================================================================
    # 4. CONCURRENCY & BATCHING
    # =========================================================================
    elif nav_mode == "Concurrency & Batching":
        st.markdown("## Concurrency & Batch Scaling Dynamics")
        st.caption("Mapping serving throughput saturation vs per-request latency inflation.")

        if not valid_results:
            st.info("Run `python scripts/benchmark.py --demo --sweep-batch` to populate data.")
            return

        df = pd.DataFrame(valid_results)
        batch_df = df.groupby("batch_size")[["tokens_per_second", "mean_latency_ms", "peak_memory_mb"]].mean().reset_index()

        col1, col2 = st.columns(2)
        with col1:
            fig_tp = px.line(
                batch_df, x="batch_size", y="tokens_per_second", markers=True,
                title="Batch Size vs Aggregate Serving Throughput (Tokens/s)",
                labels={"batch_size": "Batch Size", "tokens_per_second": "Tokens / Sec"},
            )
            fig_tp.update_traces(line_color="#10b981", marker=dict(size=8, color="#34d399", line=dict(width=2, color="#ffffff")))
            st.plotly_chart(apply_premium_chart_theme(fig_tp), use_container_width=True)

        with col2:
            fig_lat = px.line(
                batch_df, x="batch_size", y="mean_latency_ms", markers=True,
                title="Batch Size vs Per-Request Latency Inflation (ms)",
                labels={"batch_size": "Batch Size", "mean_latency_ms": "Latency (ms)"},
            )
            fig_lat.update_traces(line_color="#6366f1", marker=dict(size=8, color="#818cf8", line=dict(width=2, color="#ffffff")))
            st.plotly_chart(apply_premium_chart_theme(fig_lat), use_container_width=True)

    # =========================================================================
    # 5. CONTEXT SCALING
    # =========================================================================
    elif nav_mode == "Context Scaling":
        st.markdown("## Context Length Impact on Memory & Prefill")
        st.caption("Self-attention tensor scaling behavior and Time-To-First-Token inflation.")

        if not valid_results:
            st.info("Run `python scripts/run_experiments.py --demo` to populate data.")
            return

        df = pd.DataFrame(valid_results)
        seq_df = df.groupby("sequence_length")[["mean_latency_ms", "peak_memory_mb"]].mean().reset_index()

        col1, col2 = st.columns(2)
        with col1:
            fig1 = px.line(seq_df, x="sequence_length", y="mean_latency_ms", markers=True, title="Sequence Length vs Latency (ms)")
            fig1.update_traces(line_color="#ec4899", marker=dict(size=8, color="#f472b6"))
            st.plotly_chart(apply_premium_chart_theme(fig1), use_container_width=True)
        with col2:
            fig2 = px.line(seq_df, x="sequence_length", y="peak_memory_mb", markers=True, title="Sequence Length vs Peak Memory (MB)")
            fig2.update_traces(line_color="#06b6d4", marker=dict(size=8, color="#22d3ee"))
            st.plotly_chart(apply_premium_chart_theme(fig2), use_container_width=True)

    # =========================================================================
    # 6. PARETO FRONTIER
    # =========================================================================
    elif nav_mode == "Pareto Frontier":
        st.markdown("## Pareto Frontier: Quality vs Serving Efficiency")
        st.caption("Non-dominated operating configurations across multi-dimensional objective space.")

        if not valid_results:
            st.info("No benchmark results available.")
            return

        df = pd.DataFrame(valid_results)
        fig = px.scatter(
            df, x="mean_latency_ms", y="quality_score",
            color="precision", size="tokens_per_second",
            hover_data=["model", "batch_size", "peak_memory_mb"],
            title="Pareto Trade-Off Space: Latency vs Model Quality (Bubble Size = Serving Throughput)",
            color_discrete_sequence=["#6366f1", "#10b981", "#f59e0b", "#06b6d4"],
        )
        st.plotly_chart(apply_premium_chart_theme(fig), use_container_width=True)

    # =========================================================================
    # 7. RECOMMENDATIONS
    # =========================================================================
    elif nav_mode == "Recommendations":
        st.markdown("## Automated Optimization Recommender")
        st.caption("Normalized scoring model balancing Latency, Throughput, Memory, and Quality.")

        objective = st.selectbox(
            "Target Operational Objective",
            ["balanced", "low_latency", "high_throughput", "low_memory", "high_quality"],
            index=0,
            format_func=lambda x: x.replace("_", " ").title(),
        )

        rec_engine = RecommendationEngine(objective=objective)
        report = rec_engine.recommend(valid_results)

        if report:
            rec = report.recommended_config
            base = report.baseline_config

            st.markdown(
                f"""
                <div class="rec-box">
                    <div class="rec-title">Optimal Deployment Configuration: {objective.upper()}</div>
                    <div style="margin-top: 10px; margin-bottom: 12px;">
                        <span class="meta-tag">MODEL: {rec.get('model')}</span>
                        <span class="meta-tag">PRECISION: {rec.get('precision').upper()}</span>
                        <span class="meta-tag">BATCH: {rec.get('batch_size')}</span>
                        <span class="meta-tag">KV CACHE: {'ENABLED' if rec.get('kv_cache') else 'DISABLED'}</span>
                        <span class="meta-tag meta-tag-highlight">SCORE: {rec.get('score', 0.0):.1f} / 100.0</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Latency vs Baseline", f"{rec.get('mean_latency_ms', 0):.1f} ms", f"{report.latency_change_pct:+.1f}%", delta_color="inverse")
            c2.metric("Throughput vs Baseline", f"{rec.get('tokens_per_second', 0):.1f} tok/s", f"{report.throughput_change_pct:+.1f}%")
            c3.metric("Memory vs Baseline", f"{rec.get('peak_memory_mb', 0):.1f} MB", f"{report.memory_change_pct:+.1f}%", delta_color="inverse")
            c4.metric("Quality vs Baseline", f"{rec.get('quality_score', 0):.1f}", f"{report.quality_change_pct:+.1f}%")

            st.markdown("### Decision Rationale")
            for r in report.reasons:
                st.markdown(f"- **{r}**")
        else:
            st.warning("Insufficient benchmark records to compute recommendation. Run benchmarks first.")

    # =========================================================================
    # 8. AUDIT LOG
    # =========================================================================
    elif nav_mode == "Audit Log":
        st.markdown("## Benchmark Audit Log & Experiment History")
        st.caption("Immutable record of empirical passes stored in `results/benchmark_results.csv`.")

        if not results:
            st.info("No recorded benchmarks found.")
            return

        df = pd.DataFrame(results)
        st.dataframe(df, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            csv_data = df.to_csv(index=False).encode("utf-8")
            st.download_button("Export Tabular CSV", data=csv_data, file_name="inferx_benchmark_results.csv", mime="text/csv")
        with c2:
            json_data = df.to_json(orient="records", indent=2)
            st.download_button("Export Structured JSON", data=json_data, file_name="inferx_benchmark_results.json", mime="application/json")


if __name__ == "__main__":
    main()
