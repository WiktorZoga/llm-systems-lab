# LLM Systems Lab

My learning project: building a GPT in PyTorch and checking how its size,
attention implementation and batch shape affect time and memory.

## What is implemented

- GPT from scratch, with naive attention and optional SDPA
- Training, single-batch overfitting and text generation
- Hugging Face GPT-2 weight import
- Synthetic time and memory benchmarks, with plots in notebooks

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

Times measured on MPS in FP32, three repeats each. SDPA uses the math backend
with autograd in our setup. JSONs are in `experiments/`; checkpoints and scratch
runs stay in ignored `artifacts/`. CUDA has not been measured yet.

## Next

Understand the results, then try selected workloads on CUDA and explore lower-level GPU code.

## Learning resources

I am currently working through:

- [How to Scale Your Model](https://jax-ml.github.io/scaling-book/)
- [Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook)
- [Andrej Karpathy's lectures](https://www.youtube.com/@AndrejKarpathy)

## Acknowledgements

This project is inspired by GPT-2, nanoGPT, and educational materials by
Andrej Karpathy.
