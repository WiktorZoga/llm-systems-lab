# LLM Systems Lab

GPT from scratch in PyTorch, with experiments on training time and memory.
Built and measured on an M4 Mac. This is a learning project, not a training framework.

## What is implemented

- GPT from scratch, with naive attention and optional SDPA
- Training, single-batch overfitting and text generation
- Gradient accumulation and optional block activation checkpointing
- Hugging Face GPT-2 weight import and logits comparison
- Synthetic time and memory benchmarks, with plots in notebooks

## Where things are

```text
src/llm_systems_lab/   model, data and config code
scripts/training/     train, overfit and plot training metrics
scripts/inference/    generate text
scripts/benchmarks/   measure time/memory and draw plots
configs/              settings for training and benchmarks
experiments/          saved results, notebooks and figures
```

Setup: `uv sync --locked`. For notebooks, use the repo's `.venv/bin/python`.
For a new training run, change `[run].name`; existing run folders and checkpoints are protected.
`gpt2_hf.toml` only contains the model, tokenizer and output settings needed for import/generation.

## Model anatomy

The model consists of token embeddings, positional embeddings, causal
self-attention, MLP blocks, residual connections, and LayerNorm.

## Parameter count

$$
N = VD + CD + L(12D^2 + 13D) + 2D
$$

$V$: vocabulary, $D$: width, $C$: context limit, $L$: layers.
Embedding and LM head share weights, so $VD$ is counted once.
Tiny baseline: 6,862,464 parameters. GPT-2 reference: 124,439,808.

## Compute model

Counting a multiply-add as two FLOPs:

$$
F_{\text{block}} \approx 24BTD^2 + 4BT^2D
$$

$$
F_{\text{lm-head}} \approx 2BTDV
$$

$$
F_{\text{forward}}
\approx
L(24BTD^2 + 4BT^2D) + 2BTDV
$$

$B$: batch size, $T$: sequence length. This counts the main matmuls, not
softmax, normalization or loss. More FLOPs do not always mean more runtime.
With our tiny model and large vocabulary, the LM head dominates this estimate.

## Experiments

- [Depth scaling](experiments/01_depth_scaling/analysis.ipynb): L=1–16, predicted FLOPs vs measured time.
- [Memory](experiments/02_memory_scaling/analysis.ipynb): doubling batch size or sequence length.
- [Naive vs SDPA](experiments/03_attention/analysis.ipynb): time and memory, including longer sequences and forward with/without autograd.
- [Training memory](experiments/05_training_memory/README.md): checkpointing stores less backward state; numerical checks and a short real training run.
- `scripts/benchmarks/roofline.py`: small GEMM and copy measurements for a hardware reference.

Depth, B/T memory and attention comparisons have three repeats on MPS in FP32.
SDPA uses the math backend with autograd in our setup; it is not a FlashAttention result.
Selected JSONs and figures are in `experiments/`; checkpoints and scratch runs stay
in ignored `artifacts/`.

### Checkpointing result

For B=4, T=128, D=256, L=8 and V=4096, checkpointing reduced saved activation
storage from **81.19 to 13.00 MiB (about 84% less)**. Loss, gradients and one AdamW
update matched the normal path at the same batch size. This is saved storage,
not peak GPU memory or a measured speedup.

![Checkpointing: saved state and recomputation](experiments/05_training_memory/figures/checkpointing.png)

A separate 200-step Shakespeare run reached validation loss **5.99**, but still
generated repeated newlines. It checks that the training pipeline works, not text quality.

## Not measured yet

CUDA benchmark/profiler scripts are prepared, but there are no CUDA results.
KV-cache files are scaffolding; cached generation is not implemented.
C++ operators and Tenstorrent kernels are a possible separate project, not a feature here.

## Learning resources

I am currently working through:

- [How to Scale Your Model](https://jax-ml.github.io/scaling-book/)
- [Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook)
- [Andrej Karpathy's lectures](https://www.youtube.com/@AndrejKarpathy)

## Acknowledgements

This project is inspired by GPT-2, nanoGPT, and educational materials by
Andrej Karpathy.
