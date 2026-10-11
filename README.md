# LLM Systems Lab

My learning project for implementing and understanding language models in PyTorch.
Starting with a GPT-style Transformer: training it, importing GPT-2 weights,
and following what happens during text generation. Local experiments run on an M4 Mac.

Recently, I’ve got a Tenstorrent Wormhole 150d chip. Soon I will be running some experiments on it!

## What is implemented

- GPT from scratch, with naive attention and optional SDPA
- Training, validation, single-batch overfitting and greedy text generation
- KV cache: prompt prefill followed by one-token decode
- Gradient accumulation and optional block activation checkpointing
- Hugging Face GPT-2 weight import with a last-token logits check

GPT-2 import uses the same implementation with GPT-2 dimensions and pretrained weights;
it is not a second independently implemented architecture.

## Where things are

```text
src/llm_systems_lab/   model, data and config code
scripts/training/     train, overfit and plot training metrics
scripts/inference/    generate text
scripts/benchmarks/   measure time/memory and draw plots
configs/              settings for training and benchmarks
experiments/          saved results, notebooks and figures
```

## Try it

```bash
uv sync --locked
uv run python scripts/data/prepare_data.py
uv run python scripts/training/train.py --config gpt_tiny_checkpointing.toml
uv run python scripts/inference/generate.py \
    --config gpt_tiny_checkpointing.toml \
    --checkpoint artifacts/runs/gpt-tiny-checkpointing-smoke/checkpoints/step_000200.pt \
    --prompt 'First Citizen:' --max-new-tokens 20 --device mps --use-cache
```

This is a short training example, not a good language model yet. Remove `--use-cache`
to recompute the prefix at every step. Cached generation must fit within the context limit.
For notebooks, use the repo's `.venv/bin/python`.
For a new training run, change `[run].name`; existing run folders and checkpoints are protected.
`gpt2_hf.toml` only contains the model, tokenizer and output settings needed for import/generation.

## Model anatomy

The model consists of token embeddings, positional embeddings, causal
self-attention, MLP blocks, residual connections, and LayerNorm.

```text
token IDs → token + position embeddings → Transformer blocks → final LayerNorm → LM head
```

Prefill runs the whole prompt through this path and stores K/V for each layer.
Decode runs only the new token through it, attending to the stored keys and values.

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

</details>

## Earlier explorations

- [Depth scaling](experiments/01_depth_scaling/analysis.ipynb): L=1–16, predicted FLOPs vs measured time.
- [Memory](experiments/02_memory_scaling/analysis.ipynb): doubling batch size or sequence length.
- [Naive vs SDPA](experiments/03_attention/analysis.ipynb): time and memory, including longer sequences and forward with/without autograd.
- [Training memory](experiments/05_training_memory/README.md): checkpointing stores less backward state; numerical checks and a short real training run.
- `scripts/benchmarks/roofline.py`: small GEMM and copy measurements for a hardware reference.

Depth, B/T memory and attention comparisons have three repeats on MPS in FP32.
SDPA uses the math backend with autograd in our setup; it is not a FlashAttention result.
Selected JSONs and figures are in `experiments/`; checkpoints and scratch runs stay
in ignored `artifacts/`.

The checkpointing case saved about 84% less activation storage at the same batch size;
that is not a peak-memory or speedup claim. The separate 200-step Shakespeare run
reached validation loss 5.99 but generated repeated newlines.

## Not measured yet

CUDA benchmark/profiler scripts are prepared, but there are no CUDA results.
Fixed-capacity caches and further architectures are future work.
There are no Tenstorrent results in this repo; hardware-specific work stays separate.

## Learning with agents

I use coding agents for implementation help, code review and experiment plumbing.
The goal is to work through the code and understand the results, not to claim every
line was written unaided. [AGENTS.md](AGENTS.md) describes how we keep the project simple.

## Learning resources

I am currently working through:

- [How to Scale Your Model](https://jax-ml.github.io/scaling-book/)
- [Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook)
- [Andrej Karpathy's lectures](https://www.youtube.com/@AndrejKarpathy)

## Acknowledgements

This project is inspired by GPT-2, nanoGPT, and educational materials by
Andrej Karpathy. The [Cache Me If You Can workshop](https://github.com/mdragulaTT/cache-me-if-you-can)
by Maciek Draguła and Nathan Maurice helped guide the prefill/decode and KV-cache work.
The workshop's TTNN code and hardware results are separate from this PyTorch implementation.
