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
- synchronized memory snapshots across one training step (MPS / CUDA counters)
- optional PyTorch SDPA alongside naive attention
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

Scaling experiments vary batch size, sequence length, width, depth, heads or
vocabulary size, one at a time. We measure parameter count, forward,
forward + backward and optimizer-step time, plus tokens/s.

Results are saved as JSON, with CSV summaries and plots.
See [configs and commands](configs/benchmarks/scaling/README.md).
Memory measurements and naive vs SDPA comparisons are the next experiments.

Experiment artifacts are currently kept locally and are not included in the repository.

## Roadmap

- Repeat controlled scaling runs and compare them with manual FLOP predictions.
- Measure model, activation, gradient and optimizer-state memory.
- Run naive vs SDPA A/B on MPS, then repeat selected workloads on CUDA.
- Later: profiling, precision and compilation experiments.
- Longer term: inference, multi-GPU scaling and lower-level GPU programming.

## Learning resources

I am currently working through:

- [How to Scale Your Model](https://jax-ml.github.io/scaling-book/)
- [Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook)
- [Andrej Karpathy's lectures](https://www.youtube.com/@AndrejKarpathy)

## Acknowledgements

This project is inspired by GPT-2, nanoGPT, and educational materials by
Andrej Karpathy.
