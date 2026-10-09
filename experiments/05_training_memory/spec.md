# Training memory: things to understand

The code uses PyTorch's non-reentrant checkpointing around each GPT block.
The controlled check uses synthetic tokens and a smaller vocabulary; the separate
Shakespeare run keeps the GPT-2 tokenizer. Neither measures checkpointing speedup.

## Interview checkpoints

> **Recall:** Why do we divide each microbatch loss by the accumulation count?
> When would that scaling be wrong?

> **Recall:** What does a checkpoint keep, and what is recomputed in backward?
> Why do parameter, gradient and AdamW-state memory remain?

> **Recall:** Why can we not sum the sizes of every saved tensor view?
> What is the difference between tensor shape and its backing storage?

> **Recall:** Why might checkpointing two tiny blocks barely change total memory
> when the vocabulary is large?

> **Recall:** Why pass the block itself to checkpoint instead of a lambda that
> captures the changing loop variable? Why check numerical results before memory?

> **Recall:** Why can very similar gradients produce different first AdamW updates
> when the gradients are near its epsilon?
