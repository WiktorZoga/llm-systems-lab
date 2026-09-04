from pathlib import Path

import torch
from torch.utils.data import DataLoader

from llm_systems_lab.data.dataset import TextDataset
from llm_systems_lab.models.gpt import GPT
from llm_systems_lab.config import load_model_config

ROOT = Path(__file__).resolve().parents[1]

config = load_model_config(ROOT / "configs" / "gpt_tiny.toml")

train_ids = torch.load(ROOT / "data" / "processed" / "shakespeare" / "train.pt")

dataset = TextDataset(
    token_ids=train_ids,
    sequence_length=128
)

loader = DataLoader(dataset=dataset, batch_size=8, shuffle=None, drop_last=True)

input_ids, targets = next(iter(loader))

print(f"train token IDs: {train_ids.shape}")
print(f"dataset length: {len(dataset)}")
print(f"input_ids shape: {input_ids.shape}")
print(f"targets shape: {targets.shape}")
print(f"input_ids dtype: {input_ids.dtype}")
print(f"targets dtype: {targets.dtype}")

model = GPT(config)

logits, loss = model(input_ids, targets)

print(f"logits shape: {logits.shape}")
print(f"loss: {loss}")