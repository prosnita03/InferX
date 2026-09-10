# InferX Experimental & Benchmarking Methodology

This document details the scientific principles, statistical formulations, and systems engineering standards governing all experiments in **InferX**.

---

## 1. Benchmarking Principles & System Invariants

Benchmarking generative deep learning models differs substantially from benchmarking traditional software. In autoregressive language model serving, execution alternates between two distinct computational regimes:
1. **Prefill Phase (Prompt Processing)**: Compute-bound matrix multiplication over the entire input context.
2. **Decoding Phase (Autoregressive Generation)**: Memory-bandwidth bound vector-matrix multiplications generating one token at a time.

InferX enforces the following methodological invariants to guarantee reproducible, scientifically valid measurements.

---

## 2. Hardware Stream Synchronization

### The Asynchronous Kernel Trap
By default, PyTorch schedules CUDA and accelerator operations asynchronously on the GPU compute queue. When Python calls `model.forward(inputs)`, the CPU dispatches the CUDA kernels to the command stream and immediately returns control to Python without waiting for GPU execution to complete.

```python
# FLAWED BENCHMARKING (times host queue submission, not GPU execution):
start = time.perf_counter()
output = model.generate(...)
latency = time.perf_counter() - start  # Measures Python overhead (~0.5ms), not inference!
```

### The InferX Solution
InferX mandates hardware stream synchronization immediately before starting and after stopping high-resolution timers:

```python
# CORRECT SYNCHRONIZED TIMING:
sync_device(device)  # torch.cuda.synchronize() or torch.mps.synchronize()
start = time.perf_counter()

output = model.generate(...)

sync_device(device)  # Block host thread until all queued kernels have finished
latency = time.perf_counter() - start
```

---

## 3. The Warm-Up Protocol

A frequent source of measurement contamination is the first inference pass ("cold start"). The initial iteration includes non-recurring overheads:
1. **CUDA Kernel Compilation**: Just-in-time cuBLAS and cuDNN kernel selection and initialization.
2. **PyTorch Caching Allocator**: Initial allocation of contiguous virtual memory pools from the GPU driver.
3. **Weight Paging**: Memory hierarchy loading from system RAM or page tables into accelerator HBM.
4. **Python Bytecode Optimization**: Module import resolution and internal cache initialization.

### InferX Protocol
Every benchmark mandates a configurable warm-up phase (default: 3 iterations). Warm-up passes execute identical input shapes through the model, and all timing records from these iterations are strictly quarantined and reported separately from steady-state statistics.

---

## 4. Latency Metrics Breakdown

InferX decomposes inference latency into actionable operational metrics:

### 4.1 Time-To-First-Token (TTFT)
The wall-clock elapsed time from when a request is dispatched until the very first generated token is output. This measures the latency of the prompt prefill pass:
$$\text{TTFT} = t_{\text{first\_token}} - t_{\text{request\_dispatched}}$$

### 4.2 Inter-Token Latency (ITL)
The time required to generate each subsequent token during the autoregressive decoding phase:
$$\text{ITL}_i = t_{\text{token}_{i}} - t_{\text{token}_{i-1}} \quad \text{for } i \in [2, N]$$

### 4.3 Statistical Percentiles
Because latency distributions in production environments exhibit heavy right-tails due to memory bus contention and thread scheduling, mean latency alone is insufficient. InferX computes:
- **P50 (Median)**: The 50th percentile of response times.
- **P90 / P95**: Upper-tail bounds reflecting user-perceived delays.
- **P99**: Worst-case outlier boundary.

---

## 5. Throughput & Concurrency Metrics

Throughput measures the aggregate processing capacity of an inference engine under concurrency:

### 5.1 Token Generation Rate
$$\text{Throughput}_{\text{tokens/s}} = \frac{\sum_{b=1}^{B} N_{\text{gen}, b}}{\Delta t_{\text{wall-clock}}}$$
where $B$ is the batch size and $N_{\text{gen}, b}$ is the count of new tokens generated for sequence $b$.

### 5.2 Request Serving Rate
$$\text{Throughput}_{\text{req/s}} = \frac{B}{\Delta t_{\text{wall-clock}}}$$

### 5.3 Batch Scaling & Saturation Point
As batch size $B$ increases:
- Throughput initially scales nearly linearly because parallel tensor operations better utilize GPU Tensor Cores (moving from memory-bound to compute-bound).
- Beyond a certain hardware **saturation point**, memory bandwidth becomes fully saturated, or KV-cache allocations trigger GPU memory fragmentation.
- Eventually, batch size scaling is bounded by out-of-memory (OOM) limits.

InferX automatically detects this saturation threshold and isolates failed configurations without aborting ongoing sweeps.

---

## 6. Memory Measurement Methodology

InferX tracks distinct memory components using PyTorch CUDA APIs and system `psutil` monitors:

| Memory Metric | Definition | Measurement Method |
| :--- | :--- | :--- |
| **Baseline Memory** | Memory utilized by OS and PyTorch runtime prior to inference. | `torch.cuda.memory_allocated()` |
| **Model Weight Memory** | Static footprint occupied by parameter weights and layer buffers. | $\sum \text{numel} \times \text{element\_size}$ |
| **Peak Inference Memory** | Maximum memory allocated during forward activations and KV-cache expansion. | `torch.cuda.max_memory_allocated()` |
| **Reserved Memory** | Total memory reserved by PyTorch's caching allocator from the driver. | `torch.cuda.memory_reserved()` |

Prior to tracking, `torch.cuda.empty_cache()` and `torch.cuda.reset_peak_memory_stats()` ensure that previous allocations do not inflate current measurements.

---

## 7. Key-Value (KV) Cache Mechanics

### Theoretical Formulation
In standard multi-head self-attention without caching:
At step $t$, computing attention requires queries $Q_t$, keys $K_{0:t}$, and values $V_{0:t}$.
Without caching, all previous tokens $0 \dots t-1$ must re-run through the projection layers:
$$\text{FLOPs}(use\_cache=False) \propto \sum_{t=1}^{N} t \approx \mathcal{O}(N^2)$$

With KV caching enabled, past keys and values are stored in memory buffers:
$$\text{FLOPs}(use\_cache=True) \propto \sum_{t=1}^{N} 1 \approx \mathcal{O}(N)$$

### Memory Trade-Off
The memory footprint of the KV-cache for a batch size $B$, context length $L$, layers $N_{\text{layers}}$, and key-value heads $N_{\text{kv}}$ is:
$$\text{Memory}_{\text{KV}} = 2 \times B \times L \times N_{\text{layers}} \times N_{\text{kv}} \times d_{\text{head}} \times \text{bytes\_per\_elem}$$

InferX provides direct empirical comparison verifying this exact speedup vs memory trade-off.

---

## 8. Sources of Measurement Variance

Even in controlled laboratory conditions, benchmarks exhibit variance due to:
1. **GPU Thermal Throttling**: When GPU temperatures cross thermal targets (e.g. $83^\circ\text{C}$), the clock frequency drops automatically.
2. **Frequency Governor**: Variable GPU/CPU boost states (`p-states`).
3. **Background OS Noise**: Context switching, memory paging, and OS daemons.
4. **Driver Version & CUDA Runtime**: Minor driver versions introduce optimization variations in kernel scheduling.

To maximize reproducibility, InferX enforces random seed initialization, standardized warmups, multiple measurement iterations, and full hardware metadata logging.
