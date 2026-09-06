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
- synthetic benchmarks for scaling experiments
- benchmark summaries and plots
- scaling experiments over batch size, sequence length, model width, depth,
  attention heads, and vocabulary size

## Model anatomy

The model consists of token embeddings, positional embeddings, causal
self-attention, MLP blocks, residual connections, and LayerNorm.

## Parameter count

For the current GPT-style architecture:

$$
N = VD + CD + L(12D^2 + 13D) + 2D
$$

where:

- $V$ - vocabulary size
- $D$ - model width
- $C$ - maximum context length
- $L$ - number of Transformer blocks

The $12D^2$ term contains the attention and MLP weight matrices.
The $13D$ term contains their biases and the two LayerNorms in each block.

The token embedding and language-model head use tied weights, so the
$V \times D$ weight matrix is counted only once.

For example, the tiny baseline contains 6,862,464 parameters, while the
GPT-2 124M reference configuration contains 124,439,808 parameters.

## Compute model

Using the convention that one multiply and one add count as two FLOPs, the
dominant matmul compute for one Transformer block is approximately:

$$
F_{\text{block}} \approx 24BTD^2 + 4BT^2D
$$

The two terms have different origins:

$$
24BTD^2
$$

comes from the QKV projection, attention output projection, and MLP, while

$$
4BT^2D
$$

comes from the two attention matrix multiplications.

The language-model head costs approximately:

$$
F_{\text{lm-head}} \approx 2BTDV
$$

Therefore, the dominant matmul FLOPs for a full forward pass are approximately:

$$
F_{\text{forward}}
\approx
L(24BTD^2 + 4BT^2D) + 2BTDV
$$

where:

- $B$ - batch size
- $T$ - input sequence length
- $D$ - model width
- $L$ - number of Transformer blocks
- $V$ - vocabulary size

These estimates count the dominant matrix multiplications only. They exclude
operations such as LayerNorm, softmax, GELU, residual additions, bias additions,
embedding addition, and cross-entropy.

Some useful expectations:

- projection and MLP compute scale approximately as $D^2$;
- the two main attention matrix multiplications scale as $T^2D$;
- increasing $T$ therefore affects both linear-in-$T$ and quadratic-in-$T$
  parts of the model;
- increasing $V$ increases embedding parameters and language-model-head
  compute linearly;
- in a tiny model with a full GPT-2 vocabulary, the language-model head can
  represent a surprisingly large fraction of total compute;
- theoretical FLOPs do not necessarily translate directly into execution time:
  utilization, memory traffic, tensor shapes, kernel implementations, and
  dispatch overhead also matter.

## Experiments

The first experiments study how model and workload dimensions affect:

- forward latency
- forward + backward latency
- full optimizer-step latency
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
- inspect operators and accelerator kernels where supported
- study dispatch and kernel-launch overhead
- identify dominant operations in training
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

- 1-GPU vs 2-GPU data-parallel training
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

- use GPT-2 124M as a more realistic reference workload
- compare tiny synthetic workloads with a GPT-2-sized architecture
- repeat selected performance experiments on NVIDIA GPUs
- investigate which conclusions survive as model size increases

## Learning resources

I am currently working through:

- [How to Scale Your Model](https://jax-ml.github.io/scaling-book/)
- [Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook)
- [Andrej Karpathy's lectures](https://www.youtube.com/@AndrejKarpathy)

## Acknowledgements

This project is inspired by GPT-2, nanoGPT, and educational materials by
Andrej Karpathy.