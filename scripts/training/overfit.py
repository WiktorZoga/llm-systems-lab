import argparse
import json
import os
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from llm_systems_lab.config import load_experiment_config
from llm_systems_lab.data.dataset import TextDataset
from llm_systems_lab.models.gpt import GPT

ROOT = Path(__file__).resolve().parents[2]


def main(args):
    config = load_experiment_config(ROOT / "configs" / args.config)

    OUTPUT_DIR = ROOT / config.run.output_dir / config.run.name

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR / "checkpoints", exist_ok=True)

    metrics_path = OUTPUT_DIR / "metrics.jsonl"
    metrics_file = metrics_path.open("w", encoding="utf-8")

    torch.manual_seed(config.train.seed)
    device = config.train.device

    train_ids = torch.load(ROOT / config.data.train_path)
    train_dataset = TextDataset(
        token_ids=train_ids,
        sequence_length=config.train.sequence_length,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.train.micro_batch_size,
        shuffle=False,
        drop_last=True,
    )

    # Capture one training batch and reuse it throughout the experiment.
    fixed_input_ids, fixed_targets = next(iter(train_loader))
    fixed_input_ids = fixed_input_ids.to(device)
    fixed_targets = fixed_targets.to(device)

    model = GPT(config.model).to(device)
    optimizer = AdamW(model.parameters(), lr=config.train.learning_rate)

    model.train()

    tokens_per_step = config.train.batch_size * config.train.sequence_length
    tokens_seen = 0

    for step in range(config.train.max_steps):
        if device == "mps":
            torch.mps.synchronize()

        step_start = time.perf_counter()
        micro_losses = []

        optimizer.zero_grad(set_to_none=True)

        for _ in range(config.train.gradient_accumulation_steps):
            _, loss = model(fixed_input_ids, fixed_targets)

            scaled_loss = loss / config.train.gradient_accumulation_steps
            scaled_loss.backward()

            micro_losses.append(scaled_loss.detach())

        optimizer.step()
        train_loss = torch.stack(micro_losses).sum().item()

        if device == "mps":
            torch.mps.synchronize()

        step_time = time.perf_counter() - step_start
        tokens_seen += tokens_per_step

        metric = {
            "step": step + 1,
            "train_loss": train_loss,
            "step_time_sec": step_time,
            "tokens/s": tokens_per_step / step_time,
            "tokens_seen": tokens_seen,
            "lr": optimizer.param_groups[0]["lr"],
        }

        metrics_file.write(json.dumps(metric) + "\n")
        metrics_file.flush()

        if step == 0 or (step + 1) % 50 == 0:
            print(f"step: {step + 1} | train loss: {train_loss:.4f}")

        if (step + 1) % config.checkpoint.interval == 0:
            checkpoint = {
                "step": step + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            checkpoint_path = (OUTPUT_DIR / "checkpoints" / f"step_{step + 1:06d}.pt")
            torch.save(checkpoint, checkpoint_path)

    metrics_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="gpt_tiny_overfit.toml",
        help="Name of a config file inside the configs/ directory.",
    )

    args = parser.parse_args()
    main(args)
