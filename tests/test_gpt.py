from dataclasses import replace
from pathlib import Path

import pytest
import torch

from llm_systems_lab.config import ModelConfig, load_experiment_config
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


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_naive_and_sdpa_match(device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")

    torch.manual_seed(2137)
    config = ModelConfig(
        vocab_size=64, context_length=16, d_model=32,
        num_layers=2, num_heads=4, dropout=0.0, attention_backend="naive",
    )
    naive = GPT(config).to(device)
    sdpa = GPT(replace(config, attention_backend="sdpa")).to(device)
    sdpa.load_state_dict(naive.state_dict())

    tokens = torch.randint(config.vocab_size, (2, 8)).to(device)
    inputs = tokens[:, :-1].contiguous()
    targets = tokens[:, 1:].contiguous()

    naive_logits, naive_loss = naive(inputs, targets)
    sdpa_logits, sdpa_loss = sdpa(inputs, targets)

    # FP32 implementations may round differently; bitwise equality is not required.
    torch.testing.assert_close(naive_logits, sdpa_logits, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(naive_loss, sdpa_loss, rtol=1e-4, atol=1e-5)

    naive_loss.backward()
    sdpa_loss.backward()
    for name, parameter in naive.named_parameters():
        assert parameter.grad is not None, name
        torch.testing.assert_close(
            parameter.grad, sdpa.get_parameter(name).grad, rtol=1e-4, atol=1e-5,
            msg=f"Gradient mismatch: {name}",
        )
