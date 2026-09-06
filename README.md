# LLM Systems Lab

A personal, from-scratch learning project about how transformer models execute,
scale, and interact with real hardware.

The goal is to understand LLMs both as neural networks and as computational
workloads: from model math and PyTorch execution to profiling, optimization,
distributed training, and eventually lower-level C++/GPU programming.

## What is implemented

- GPT-style Transformer implemented in PyTorch
- training and overfitting scripts
- text generation
- GPT-2 weight import
- synthetic benchmark for scaling experiments
- benchmark summaries and plots
- scaling experiments over batch size, sequence length, model width, depth,
  attention heads, and vocabulary size

## Model anatomy

The model consists of token embeddings, positional embeddings, causal
self-attention, MLP blocks, residual connections, and LayerNorm.

## Parameter count

For the current GPT-style architecture:

$$
N \approx VD + CD + L(12D^2 + 13D) + 2D
$$

The token embedding and language-model head use tied weights, so the vocabulary
projection weights are counted only once.

Symbols:

- $V$ - vocabulary size
- $D$ - model width
- $C$ - maximum context length
- $L$ - number of Transformer blocks

## Compute model

Approximate matmul compute for one Transformer block:

$$
F_{\text{block}} \approx 24BTD^2 + 4BT^2D
$$

Approximate compute for the language-model head:

$$
F_{\text{lm-head}} \approx 2BTDV
$$

where:

- $B$ - batch size
- $T$ - input sequence length
- $D$ - model width
- $V$ - vocabulary size

Some useful expectations:

- increasing $D$ increases most Transformer block compute roughly quadratically;
- increasing $T$ increases linear projections linearly, while attention contains
  a quadratic $T^2$ term;
- increasing $V$ increases both embedding parameters and vocabulary-projection
  compute linearly;
- theoretical FLOPs do not necessarily translate directly into execution time
  because hardware utilization, memory traffic, kernel shapes, and overhead also matter.

## Experiments

The first experiments study how model and workload dimensions affect:

- forward latency
- forward + backward latency
- optimizer-step latency
- tokens / second
- parameter count

Raw benchmark artifacts are generated locally. Selected results and conclusions
will be added here as the experiments mature.

## Roadmap

### 1. Scaling and analytical performance model

- parameter counting
- forward FLOPs estimation
- model-state memory accounting
- batch-size scaling
- sequence-length scaling
- model-width scaling
- depth scaling
- attention-head scaling
- vocabulary-size scaling
- compare predicted compute with measured runtime

### 2. Profiling

- learn the PyTorch profiler
- understand CPU vs device time
- inspect operators and accelerator kernels
- study kernel launch overhead
- identify the dominant operations in training
- compare profiler results with the analytical FLOP model
- investigate memory usage and temporary tensors

### 3. Single-device optimization

- naive attention vs PyTorch SDPA
- study performance across different sequence lengths
- FP32 vs BF16
- `torch.compile`
- operator fusion and reduced intermediate materialization
- measure speed, memory, and correctness after every optimization

### 4. Inference systems

- separate prefill and autoregressive decode
- implement KV caching
- measure TTFT and inter-token latency
- benchmark throughput across prompt lengths and batch sizes
- study the memory cost of the KV cache
- experiment with MHA, GQA, and MQA

### 5. Multi-GPU systems

- 1-GPU vs 2-GPU data parallel training
- measure scaling efficiency
- inspect NCCL communication
- understand gradient AllReduce
- implement a small tensor-parallel MLP manually
- experiment with AllGather, ReduceScatter, and AllReduce
- compare predicted communication cost with measured runtime

### 6. Lower-level performance engineering

- revisit cache locality and memory layout in C++
- implement naive and tiled CPU matrix multiplication
- study blocking, data reuse, and arithmetic intensity
- experiment with PyTorch C++ extensions
- study GPU kernel execution and memory hierarchy
- eventually implement a small CUDA or Triton kernel
- connect low-level measurements back to Transformer workloads

### 7. Larger reference workloads

- use GPT-2 124M as a more realistic reference model
- compare tiny synthetic workloads with a real GPT-2-sized architecture
- repeat selected performance experiments on NVIDIA GPUs
- investigate which conclusions survive as model size increases

## Educational Material
Currently I'm slowly selfstudying from these materials:
    - https://jax-ml.github.io/scaling-book/
    - https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=high-level_overview
    - https://www.youtube.com/@AndrejKarpathy
    - more soon

## Acknowledgements

This project is inspired by GPT-2, nanoGPT, and educational materials by
Andrej Karpathy.