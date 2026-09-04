"""
    GPT-2 model architecture.
"""
import math

import torch
import torch.nn.functional as F
from torch import nn

from llm_systems_lab.config import ModelConfig

class MultiHeadSelfAttention(nn.Module):
    """Causal multi-head self-attention used inside a Transformer block."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        self.config = config

        # One projection produces the concatenated query, key, and value tensors.
        self.c_attn = nn.Linear(in_features=config.d_model, out_features=3*config.d_model)

        # Mix information from all attention heads back into d_model dimensions.
        self.c_proj = nn.Linear(in_features=config.d_model, out_features=config.d_model)

        # Lower-triangular mask: a token may attend to itself and previous tokens,
        # but never to tokens that occur later in the sequence.
        self.bias = nn.Buffer(torch.tril(torch.ones(config.context_length, config.context_length)).view(1, 1, config.context_length, config.context_length))

    def forward(self, x):
        """
            x: [B, T, C] - batch_size, sequence_length, embedding_dim

            q, k, v: [B, nh, T, hd] - batch_size, number_of_heads, sequence_length, head_dim

            att: [B, nh, T, T]

            y: [B, nh, T, hd] -> [B, T, C]
        """
        B, T, C = x.size()

        # Split the combined projection into independent Q, K, and V tensors.
        q, k, v = self.c_attn(x).split(self.config.d_model, dim=2)

        head_dim = C // self.config.num_heads

        # Move the head dimension before the sequence dimension so attention
        # can be calculated independently for every head.
        q = q.view(B, T, self.config.num_heads, head_dim).transpose(1, 2)
        k = k.view(B, T, self.config.num_heads, head_dim).transpose(1, 2)
        v = v.view(B, T, self.config.num_heads, head_dim).transpose(1, 2)

        # Scaled dot-product attention produces one score for every pair of
        # query/key positions: [B, num_heads, T, T].
        att = (q @ k.mT) * (1.0 / math.sqrt(head_dim))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)

        # Weight values by the attention scores, then merge all heads back
        # into the original representation shape [B, T, C].
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)

        y = self.c_proj(y)

        return y

class MLP(nn.Module):
    """Position-wise feed-forward network used inside a Transformer block."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        self.config = config

        # Expand the representation before applying the non-linearity.
        self.c_fc = nn.Linear(in_features=config.d_model, out_features=4*config.d_model)

        self.act = nn.GELU(approximate="tanh")

        # Project the expanded representation back to d_model dimensions.
        self.c_proj = nn.Linear(in_features=4*config.d_model, out_features=config.d_model)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    """Pre-normalized Transformer block with attention and MLP residual paths."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        self.config = config

        self.ln_1 = nn.LayerNorm(normalized_shape=config.d_model)

        self.attn = MultiHeadSelfAttention(config)

        self.mlp = MLP(config)

        self.ln_2 = nn.LayerNorm(normalized_shape=config.d_model)

    def forward(self, x):
        # Normalize before each sublayer, then preserve a residual connection.
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    """Small GPT-style language model composed of stacked Transformer blocks."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        self.config = config

        self.transformer = nn.ModuleDict({
            # Token and position embeddings both produce [B, T, d_model]
            # after broadcasting the position embeddings across the batch.
            "wte": nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.d_model),
            "wpe": nn.Embedding(num_embeddings=config.context_length, embedding_dim=config.d_model),
            # ModuleList registers every Transformer block with PyTorch.
            "h": nn.ModuleList([Block(config) for _ in range(config.num_layers)]),
            "ln_f": nn.LayerNorm(normalized_shape=config.d_model)
        })

        self.lm_head = nn.Linear(in_features=config.d_model, out_features=config.vocab_size, bias=False)

        # Weight tying: use the token embedding matrix again for the final
        # vocabulary projection instead of learning a second matrix.
        self.lm_head.weight = self.transformer["wte"].weight

    def forward(self, input_ids, targets=None):
        """
            B, T - batch_size, sequence_length

            input:
                input_ids: [B, T]
                targets: [B, T]
            
            output:
                logits:    [B, T, vocab_size]
        """

        device = input_ids.device

        _, T = input_ids.size()

        assert T <= self.config.context_length

        pos = torch.arange(0, T, dtype=torch.long, device=device)

        tok_emb = self.transformer["wte"](input_ids)
        pos_emb = self.transformer["wpe"](pos)

        # Broadcasting adds the same position embedding sequence to every
        # example in the batch.
        x = tok_emb + pos_emb

        for block in self.transformer["h"]:
            x = block(x)

        x = self.transformer["ln_f"](x)

        if targets is not None:
            # During training we need logits for every position in the sequence.
            logits = self.lm_head(x)    
            loss = F.cross_entropy(input=logits.view(-1, self.config.vocab_size), target=targets.view(-1)) 
        else:
            # During inference we only need the prediction for the final token.
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss
