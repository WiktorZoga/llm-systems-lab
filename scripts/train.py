from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from llm_systems_lab.config import load_experiment_config
from llm_systems_lab.data.dataset import TextDataset
from llm_systems_lab.models.gpt import GPT

ROOT = Path(__file__).resolve().parents[1]

def main():
    config = load_experiment_config(ROOT / "configs" / "gpt_tiny_smoke.toml")

    torch.manual_seed(config.train.seed)

    train_ids = torch.load(ROOT / config.data.train_path)

    train_dataset = TextDataset(token_ids=train_ids, sequence_length=config.train.sequence_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train.micro_batch_size,
        shuffle=True,
        drop_last=True,
    )

    device = config.train.device

    model = GPT(config.model).to(device)

    optimizer = AdamW(model.parameters(), lr=config.train.learning_rate)

    model.train()

    train_iterator = iter(train_loader)

    for step in range(config.train.max_steps):

        micro_losses = []

        optimizer.zero_grad()

        for micro_step in range(config.train.gradient_accumulation_steps):
            try:
                input_ids, targets = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_loader)
                input_ids, targets = next(train_iterator)

            input_ids = input_ids.to(device)
            targets = targets.to(device)

            _, loss = model(input_ids, targets)

            scaled_loss = loss / config.train.gradient_accumulation_steps
            scaled_loss.backward()

            micro_losses.append(scaled_loss.detach())

        optimizer.step()

        step_loss = torch.stack(micro_losses).sum().item()
        print(f"step {step}: loss {step_loss:.4f}")

if __name__ == "__main__":
    main()