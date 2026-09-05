from pathlib import Path

import torch

from llm_systems_lab.config import load_experiment_config
from llm_systems_lab.models.gpt import GPT

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "gpt_tiny.toml"

def test_gpt_forward_without_targets() -> None:
    config = load_experiment_config(CONFIG_PATH)
    model = GPT(config.model)

    input_ids = torch.randint(
        low=0,
        high=config.model.vocab_size,
        size=(2, 8),
    )

    logits, loss = model(input_ids)

    assert logits.shape == (2, 1, config.model.vocab_size)
    assert loss is None

def test_gpt_forward_with_targets() -> None:
    config = load_experiment_config(CONFIG_PATH)
    model = GPT(config.model)

    input_ids = torch.randint(
        low=0,
        high=config.model.vocab_size,
        size=(2, 8),
    )

    targets = torch.randint(
        low=0,
        high=config.model.vocab_size,
        size=(2, 8),
    )

    logits, loss = model(input_ids, targets)

    assert logits.shape == (2, 8, config.model.vocab_size)
    assert loss is not None
    assert loss.ndim == 0
    assert torch.isfinite(loss)