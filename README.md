# InferX — LLM Inference Optimization & Benchmarking Platform

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/Transformers-4.38%2B-yellow.svg)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red.svg)](https://streamlit.io/)

> **A production-grade platform for measuring, optimizing, and evaluating causal language model inference performance across precisions (FP32, FP16, INT8, INT4), batch sizes, sequence lengths, and KV-cache architectures.**

Built specifically to demonstrate core competencies for an **ML Systems / Generative AI Intern & Engineer role**: PyTorch internals, CUDA stream synchronization, memory hierarchy tracking, high-throughput batch serving, multi-task NLP evaluation, and automated multi-objective Pareto optimization.

---

## 1. Problem Statement & Why Inference Optimization Matters

In generative AI deployments, running Large Language Models (LLMs) at scale is computationally expensive and memory-bandwidth bound:
- **Serving Costs**: Over 80% of an LLM's operational lifecycle budget is spent on inference, not training.
- **Latency Constraints**: Interactive applications require strict Time-To-First-Token (TTFT < 200ms) and high generation speeds (> 30 tokens/sec).
- **VRAM Bottlenecks**: Hardware memory limits determine maximum concurrency, model parameter capacity, and key-value cache expansion.

**InferX** provides an end-to-end scientific testbed to measure, dissect, and optimize these competing trade-offs under rigorous experimental controls.

---

## 2. Project Architecture

```mermaid
flowchart TD
    CLI[CLI Scripts / User] --> Config[YAML Configuration Layer]
    Dashboard[Streamlit Dashboard] --> Config
    Config --> Runner[Experiment Runner]
    
    subgraph Core ["InferX Core (src/inferx)"]
        Runner --> Loader[ModelLoader & ModelInfo]
        Loader --> Engine[InferenceEngine]
        Engine --> Cache[KV Cache Manager]
        Engine --> Batching[Left-Padding Batch Pipeline]
        
        Runner --> LatBench[Latency Benchmark (TTFT, ITL, P95)]
        Runner --> Throughput[Throughput Benchmark (tok/s, req/s)]
        Runner --> Memory[Memory Tracker (VRAM / RSS)]
        Runner --> Evaluator[Multi-Task Evaluator (QA, Sum, F1)]
        
        Runner --> Scoring[Multi-Objective Scoring Engine]
        Runner --> Recommender[Pareto Recommendation Engine]
    end
    
    Runner --> Storage[(Results: CSV & JSON)]
    Storage --> Visualizer[Plot Generator (11 Visualizations)]
    Visualizer --> Dashboard
    Storage --> Dashboard
```

---

## 3. Technology Stack

- **Core Runtime**: Python 3.10+, PyTorch 2.0+, Hugging Face Transformers, Accelerate, NumPy, Pandas
- **Optimization & Quantization**: bitsandbytes (INT8/INT4 on CUDA), PyTorch Native Dynamic Quantization (INT8 on CPU), KV Caching, Left-padded Batched Generation
- **Profiling & Hardware**: CUDA Runtime APIs, PyTorch Profiler, `psutil`, `nvidia-smi` integration
- **Evaluation**: ROUGE (`rouge-score`), Token F1, Exact Match (EM), Cross-Entropy Perplexity, Deterministic Consistency
- **Visualization & UI**: Matplotlib, Plotly, Streamlit
- **Testing**: pytest (19 automated unit tests covering loader, benchmarks, metrics, and optimizer)

---

## 4. Hardware Support & Auto-Detection

InferX automatically profiles and adapts to available compute hardware:
- **NVIDIA GPU + CUDA**: Automatically utilizes Tensor Cores, half-precision (`fp16`/`bf16`), `bitsandbytes` (INT8/INT4), and CUDA memory tracking APIs.
- **Apple Silicon (MPS)**: Utilizes Metal Performance Shaders for accelerated local evaluation on macOS.
- **CPU Fallback**: Gracefully operates in CPU-only environments with PyTorch dynamic quantization and `psutil` process memory tracking.

*Note: InferX never silently fabricates numbers or downcasts precisions. If an unsupported configuration (e.g. INT4 on CPU) is requested, it logs an explicit `HardwareNotSupportedError`.*

---

## 5. Quickstart & Installation

### Step 1: Clone and Set Up Environment
```bash
git clone https://github.com/your-username/InferX.git
cd InferX

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### Step 2: Install Dependencies
```bash
# For standard CPU / Apple Silicon environments:
pip install -r requirements.txt

# For NVIDIA GPU / CUDA environments (adds bitsandbytes):
pip install -r requirements-gpu.txt
```

---

## 6. How to Run InferX

### 1. Run Text Generation Inference
```bash
# Instant demo run using lightweight micro-model:
python scripts/run_inference.py --demo

# Full causal model inference with token-by-token TTFT profiling:
python scripts/run_inference.py \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --prompt "Explain the difference between latency and throughput in distributed systems." \
    --precision fp16 \
    --measure-tokens
```

### 2. Run Comprehensive Benchmarks
```bash
# Instant demo benchmark (measures real latency, throughput, and memory on your current machine):
python scripts/benchmark.py --demo

# Full benchmark with batch throughput sweep and KV-cache comparison:
python scripts/benchmark.py \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --precision fp16 \
    --batch-size 4 \
    --sweep-batch \
    --test-kv-cache
```

### 3. Run Multi-Task Quality Evaluation
```bash
# Evaluate quality on QA, factual recall, reasoning, and summarization:
python scripts/evaluate.py --demo
```

### 4. Run Automated Experiment Matrix
```bash
# Execute combinatorial sweep and output Pareto optimization recommendations:
python scripts/run_experiments.py --demo --objective balanced
```

### 5. Profile PyTorch Operators (Low-Level CUDA/CPU Kernels)
```bash
python scripts/profile.py --demo
```

### 6. Launch the Interactive Streamlit Dashboard
```bash
streamlit run dashboard/app.py
```

---

## 7. Core Concepts Explained

### 7.1 Quantization (FP16 vs INT8 vs INT4)
- **FP16 (Half Precision)**: Standard 16-bit float (1 sign, 5 exponent, 10 mantissa). Requires 2 bytes per parameter.
- **INT8 (8-bit Quantization)**: Compresses weights to 1 byte per parameter using vector-wise scaling. Achieves ~50% VRAM reduction with negligible perplexity loss.
- **INT4 (NormalFloat4 / FP4)**: Quantizes weights to 0.5 bytes per parameter. Drastically reduces memory footprints, enabling 7B+ models to run on consumer GPUs, at the cost of slight dequantization overhead on small batch sizes.

### 7.2 Key-Value (KV) Caching
In autoregressive transformer generation, generating token $t$ requires computing self-attention over tokens $0 \dots t-1$:
- **Without Cache (`use_cache=False`)**: Keys and values for all past tokens are re-projected at every single step, incurring quadratic $\mathcal{O}(N^2)$ computational complexity.
- **With Cache (`use_cache=True`)**: Past keys and values are retained in memory. Only the newest token is projected, reducing per-token decoding to linear $\mathcal{O}(N)$ complexity at the expense of dynamic VRAM usage.

### 7.3 Batching & Throughput Saturation
Small batch sizes ($B=1$) under-utilize modern GPU Tensor Cores, leaving memory bandwidth under-saturated. Increasing batch size groups matrix multiplications into denser GEMM operations, substantially increasing tokens/second until reaching memory bandwidth saturation or triggering Out-Of-Memory (OOM) limits.

---

## 8. Multi-Objective Scoring & Pareto Optimization

InferX implements a normalized multi-objective scoring formula:
$$\text{Score} = \frac{w_q \cdot Q_{\text{norm}} + w_t \cdot T_{\text{norm}} + w_l \cdot L_{\text{norm}} + w_m \cdot M_{\text{norm}}}{w_q + w_t + w_l + w_m} \times 100$$
*(where Latency $L$ and Memory $M$ are inverted during Min-Max normalization).*

Users can select from 5 operational presets:
1. **Low Latency**: $w_{\text{lat}}=0.60, w_{\text{thr}}=0.10, w_{\text{mem}}=0.10, w_{\text{qua}}=0.20$
2. **High Throughput**: $w_{\text{thr}}=0.60, w_{\text{lat}}=0.10, w_{\text{mem}}=0.10, w_{\text{qua}}=0.20$
3. **Low Memory**: $w_{\text{mem}}=0.60, w_{\text{lat}}=0.10, w_{\text{thr}}=0.10, w_{\text{qua}}=0.20$
4. **High Quality**: $w_{\text{qua}}=0.70, w_{\text{lat}}=0.10, w_{\text{thr}}=0.10, w_{\text{mem}}=0.10$
5. **Balanced**: $w_{\text{qua}}=0.35, w_{\text{thr}}=0.25, w_{\text{lat}}=0.25, w_{\text{mem}}=0.15$

The **Pareto Optimizer** eliminates dominated configurations, guaranteeing that recommended settings provide non-dominated efficiency.

---

## 9. Benchmark Results & Hardware Profiles

> [!NOTE]
> **No Fake Data Principle**: InferX does not fabricate synthetic benchmark values. Run the benchmark suite on your target hardware to populate this section with empirical results.

```bash
# Populate results on your system:
python scripts/run_experiments.py --demo
```

*Results are automatically logged to `results/benchmark_results.csv` and `results/benchmark_results.json`, with 11 publication-quality plots saved in `results/plots/`.*

---

## 10. Repository Structure

```
InferX/
├── README.md                 # Master project documentation & presentation guide
├── LICENSE                   # MIT License
├── pyproject.toml            # PEP 621 packaging and test configuration
├── requirements.txt          # Core dependencies
├── requirements-gpu.txt      # CUDA-specific dependencies
│
├── configs/                  # Modular configuration specifications
│   ├── models.yaml           # Model registry and parameter specifications
│   ├── benchmark.yaml        # Matrix parameters and scoring objective weights
│   └── hardware.yaml         # Device priorities and memory thresholds
│
├── src/inferx/               # Core library implementation
│   ├── models/               # ModelLoader, ModelInfo, QuantizationManager
│   ├── inference/            # InferenceEngine, BatchManager, KVCacheComparison
│   ├── benchmark/            # Latency, Throughput, MemoryTracker, PyTorchProfiler
│   ├── evaluation/           # EvaluationDataset, Evaluator, Task-Aware Metrics
│   ├── optimization/         # ScoringEngine, ParetoOptimizer, RecommendationEngine
│   ├── experiments/          # Automated ExperimentRunner & Persistence
│   └── utils/                # Config loader, Structured Logger, System Timers
│
├── scripts/                  # Production CLI interfaces
│   ├── run_inference.py      # Interactive text-generation CLI
│   ├── benchmark.py          # Latency, memory, and throughput benchmark CLI
│   ├── evaluate.py           # Multi-task quality evaluation CLI
│   ├── run_experiments.py    # Batch matrix execution CLI
│   └── profile.py            # PyTorch operator profiling CLI
│
├── dashboard/
│   └── app.py                # Interactive Streamlit dashboard
│
├── tests/                    # Automated unit tests (19 tests)
│   ├── test_loader.py
│   ├── test_benchmark.py
│   ├── test_metrics.py
│   └── test_optimizer.py
│
├── data/
│   ├── README.md             # Dataset documentation
│   └── eval_dataset.json     # Multi-task evaluation prompts
│
├── results/                  # Structured benchmark outputs
│   ├── benchmark_results.csv # Tabular benchmark log
│   ├── benchmark_results.json# Detailed JSON records
│   ├── plots/                # 11 automatically generated plots
│   └── traces/               # PyTorch Chrome profiler traces
│
└── docs/                     # Technical deep-dive documentation
    ├── architecture.md       # Detailed subsystem design
    ├── methodology.md        # Scientific benchmarking standards
    └── experiments.md        # Hypotheses and analysis guides
```

---

## 11. Automated Test Suite

Run the full pytest suite:
```bash
pytest tests/ -v
```

All 19 unit tests execute in seconds using lightweight mock models, validating configuration validation, device detection, percentile statistics, task metrics, and Pareto optimization.

---

## 12. 3-Minute Demo Flow for ML Interviewers

When presenting InferX to an ML engineering interviewer:

1. **Minute 1: The Problem & Architecture (Architecture & Systems Thinking)**
   - Open `dashboard/app.py` or `docs/architecture.md`.
   - Explain why naive LLM benchmarking fails: asynchronous CUDA kernel dispatch, cold-start memory jitter, and the prefill vs decode distinction.
   - Highlight InferX's clean modular separation between model loading, hardware stream synchronization, memory tracking, and multi-task evaluation.

2. **Minute 2: Live Empirical Demonstration (Hands-on Systems Engineering)**
   - Run `python scripts/benchmark.py --demo --test-kv-cache --sweep-batch`.
   - Walk the interviewer through the console output:
     - Point out the **empirical KV-cache speedup ratio** ($\approx 2-3\times$) and explain the $O(N^2)$ vs $O(N)$ algorithmic complexity difference.
     - Show the **batch throughput scaling curve** from Batch 1 to Batch 4 and explain Tensor Core utilization and memory bandwidth saturation.

3. **Minute 3: Pareto Frontier & Automated Recommendation (Applied ML Optimization)**
   - Navigate to the **Optimization & Recommender** tab in the Streamlit dashboard or run `python scripts/run_experiments.py --demo --objective high_throughput`.
   - Show how InferX balances conflicting operational objectives (latency vs throughput vs memory vs quality) using Min-Max normalization.
   - Explain the generated data-backed recommendation card showing exact percentage improvements relative to baseline.
