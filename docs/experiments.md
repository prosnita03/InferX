# InferX Controlled Experiments Guide

This document defines the scientific hypotheses, experimental designs, independent and dependent variables, and analytical frameworks across all **InferX** optimization studies.

---

## Experiment 1: Precision Quantization Trade-Offs (FP16 vs INT8 vs INT4)

### Scientific Hypothesis
Lower-bit quantization (INT8 and INT4) reduces GPU memory footprint roughly proportional to bit-width reduction ($\approx 50\%$ for INT8, $\approx 75\%$ for INT4 relative to FP16). However, latency does not necessarily scale linearly with bit-reduction because 4-bit dequantization overhead on compute-bound Tensor Cores can introduce runtime penalties on small batch sizes. Output quality degradation will be minimal on INT8 ($\le 1.5\%$) but more pronounced on INT4 ($\ge 3\%$) for complex reasoning tasks.

### Experimental Design
- **Independent Variable**: Precision configuration (`fp16`, `int8`, `int4`).
- **Controlled Variables**: Model architecture (`TinyLlama-1.1B`), prompt sequence length ($128$), generation token budget ($64$), batch size ($1$), sampling mode (greedy deterministic), device thermal profile.
- **Dependent Variables (Measurements)**:
  1. Parameter weight memory footprint (MB)
  2. Peak inference memory (MB)
  3. Mean steady-state latency (ms)
  4. Serving throughput (tok/s)
  5. Multi-task quality score ($0-100$)

### Analytical Formulas
$$\text{Memory Reduction \%} = \frac{\text{Baseline}_{\text{FP16}} - \text{Memory}_{\text{Quant}}}{\text{Baseline}_{\text{FP16}}} \times 100$$
$$\text{Throughput Improvement \%} = \frac{\text{Throughput}_{\text{Quant}} - \text{Throughput}_{\text{FP16}}}{\text{Throughput}_{\text{FP16}}} \times 100$$

---

## Experiment 2: Batch Size & Concurrency Scaling

### Scientific Hypothesis
Increasing batch size from $1$ to $8$ increases aggregate serving throughput (tokens/sec) substantially by shifting execution from memory-bandwidth bound to compute-bound GPU utilization. However, per-request latency will increase monotonically as batch size grows, reaching a hardware saturation threshold where further batching yields diminishing throughput returns and risks Out-Of-Memory (OOM) failures.

### Experimental Design
- **Independent Variable**: Batch size $B \in \{1, 2, 4, 8, 16\}$.
- **Controlled Variables**: Precision (`fp16`), sequence length ($128$), max new tokens ($64$), KV cache enabled (`True`).
- **Dependent Variables**:
  1. End-to-end request latency (ms)
  2. Aggregate generation throughput (tok/s)
  3. Peak memory utilization (MB)
  4. OOM boundary detection

### Analysis: Saturation Point Identification
The saturation point is identified as the batch size $B^*$ where:
$$\frac{\partial \text{Throughput}}{\partial B} \approx 0 \quad \text{or} \quad \frac{\Delta \text{Throughput}}{\text{Throughput}} < 5\%$$

---

## Experiment 3: Context Sequence Length Impact

### Scientific Hypothesis
Increasing input prompt length linearly increases memory consumption during prompt prefill and quadratically inflates attention matrix allocations if flash attention is not utilized. Longer sequence lengths proportionally increase Time-To-First-Token (TTFT) while having minimal effect on per-token decode latency (ITL).

### Experimental Design
- **Independent Variable**: Sequence length $L \in \{128, 256, 512, 1024\}$.
- **Controlled Variables**: Model (`TinyLlama-1.1B`), batch size ($1$), precision (`fp16`), max new tokens ($32$).
- **Dependent Variables**:
  1. Time-To-First-Token (TTFT in ms)
  2. Peak prefill memory (MB)
  3. Mean Inter-Token Latency (ITL in ms)

---

## Experiment 4: Autoregressive KV-Cache Ablation

### Scientific Hypothesis
Disabling the Key-Value cache (`use_cache=False`) requires quadratic $\mathcal{O}(N^2)$ recomputation of self-attention matrices at every generation step $t$, leading to an order-of-magnitude increase in total decode latency compared to linear $\mathcal{O}(N)$ cached execution (`use_cache=True`). Conversely, enabling the cache consumes additional VRAM proportional to batch size, sequence length, and model layer count.

### Experimental Design
- **Independent Variable**: KV-cache state (`use_cache=True` vs `use_cache=False`).
- **Controlled Variables**: Prompt text, generated token count ($64$), precision (`fp32`/`fp16`), batch size ($1$).
- **Dependent Variables**:
  1. Total decode latency (ms)
  2. Tokens generated per second
  3. Memory overhead ($\Delta \text{MB}$)

### Empirical Speedup Metric
$$\text{Speedup Ratio} = \frac{\text{Latency}(use\_cache=False)}{\text{Latency}(use\_cache=True)}$$

---

## Experiment 5: Multi-Task Quality vs Efficiency Pareto Frontier

### Scientific Hypothesis
No single inference configuration dominates across all operational criteria. Evaluating configurations across a 4-dimensional objective space (Latency, Throughput, Memory, Quality) reveals a distinct **Pareto frontier** of non-dominated operating points tailored for varying production deployment constraints (e.g. edge device vs high-concurrency cloud serving).

### Analysis: Pareto Dominance Condition
A candidate configuration $C$ is dominated by $O$ if and only if:
$$\forall m \in \{\text{Quality}, \text{Throughput}, -\text{Latency}, -\text{Memory}\}: O_m \ge C_m$$
$$\text{and } \exists m: O_m > C_m$$
Configurations satisfying $\neg \text{Dominated}(C)$ form the empirical Pareto frontier.
