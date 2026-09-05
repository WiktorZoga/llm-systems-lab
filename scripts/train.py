from pathlib import Path
import time
import os
import json

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from llm_systems_lab.config import load_experiment_config
from llm_systems_lab.data.dataset import TextDataset
from llm_systems_lab.data.dataloader import InfiniteDataLoader
from llm_systems_lab.models.gpt import GPT

ROOT = Path(__file__).resolve().parents[1]

def main():
    config = load_experiment_config(ROOT / "configs" / "gpt_tiny_smoke.toml")

    OUTPUT_DIR = ROOT / config.run.output_dir / config.run.name

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR / "checkpoints", exist_ok=True)

    metrics_path = OUTPUT_DIR / "metrics.jsonl"
    metrics_file = metrics_path.open("w", encoding="utf-8")

    torch.manual_seed(config.train.seed)
    device = config.train.device

    train_ids = torch.load(ROOT / config.data.train_path)
    val_ids = torch.load(ROOT / config.data.val_path)

    train_dataset = TextDataset(token_ids=train_ids, sequence_length=config.train.sequence_length)
    val_dataset = TextDataset(token_ids=val_ids, sequence_length=config.train.sequence_length)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.train.micro_batch_size,
        shuffle=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.train.micro_batch_size,
        shuffle=False,
        drop_last=True
    )

    data_loader = InfiniteDataLoader(train_loader)


    model = GPT(config.model).to(device)

    optimizer = AdamW(model.parameters(), lr=config.train.learning_rate)

    model.train()

    train_iterator = iter(data_loader)

    tokens_per_step = config.train.batch_size * config.train.sequence_length
    tokens_seen = 0

    for step in range(config.train.max_steps):

        if device == "mps":
            torch.mps.synchronize()

        step_start = time.perf_counter()

        micro_losses = []

        optimizer.zero_grad(set_to_none=True)

        for micro_step in range(config.train.gradient_accumulation_steps):

            input_ids, targets = next(train_iterator)

            input_ids = input_ids.to(device)
            targets = targets.to(device)

            _, loss = model(input_ids, targets)

            scaled_loss = loss / config.train.gradient_accumulation_steps
            scaled_loss.backward()

            micro_losses.append(scaled_loss.detach())

        optimizer.step()
        train_loss = torch.stack(micro_losses).sum().item()

        if device == "mps":
            torch.mps.synchronize()

        dt = time.perf_counter() - step_start
        tokens_per_second = tokens_per_step / dt
        tokens_seen += tokens_per_step

        val_loss = None

        if (step + 1) % config.eval.interval == 0:
            with torch.no_grad():
                model.eval()

                val_losses = []

                for eval_step, (input_ids, targets) in enumerate(val_loader):
                    if eval_step >= config.eval.num_batches:
                        break

                    input_ids = input_ids.to(device)
                    targets = targets.to(device)

                    _, loss = model(input_ids, targets)

                    if loss is None:
                        raise RuntimeError("Expected validation loss.")

                    val_losses.append(loss.detach())

                val_loss = torch.stack(val_losses).mean().item()
                model.train()
            print(f"step: {step + 1} | train loss: {train_loss:.4f} | val loss: {val_loss:.4f} | lr: {optimizer.param_groups[0]['lr']:.6f} | dt: {dt:.3f} | tokens/s: {tokens_per_second:.2f} | tokens seen: {tokens_seen}")
        else: 
            print(f"step: {step + 1} | train loss: {train_loss:.4f} | lr: {optimizer.param_groups[0]['lr']:.6f} | dt: {dt:.3f} | tokens/s: {tokens_per_second:.2f} | tokens seen: {tokens_seen}")


        if (step + 1) % config.checkpoint.interval == 0:
            checkpoint = {
                "step": step + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            checkpoint_path = OUTPUT_DIR / "checkpoints" / f"step_{step + 1:06d}.pt"
            torch.save(checkpoint, checkpoint_path)

        metric = {
            "step": step + 1,
            "train_loss": train_loss,
            "step_time_sec": dt,
            "tokens/s": tokens_per_second,
            "tokens_seen": tokens_seen,
        }

        if val_loss is not None:
            metric["val_loss"] = val_loss

        metrics_file.write(json.dumps(metric) + "\n")
        metrics_file.flush()

if __name__ == "__main__":
    main()