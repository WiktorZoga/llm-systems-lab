"""Create fresh runs and keep checkpoint files from being overwritten."""

from pathlib import Path

import torch


def create_run_directory(path):
    path = Path(path)
    # mkdir fails if this run already exists; choose a new run name.
    path.mkdir(parents=True)
    (path / "checkpoints").mkdir()
    return path


def save_checkpoint(checkpoint, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # "x" creates a new file only; "b" lets torch write binary data.
    with path.open("xb") as file:
        torch.save(checkpoint, file)
