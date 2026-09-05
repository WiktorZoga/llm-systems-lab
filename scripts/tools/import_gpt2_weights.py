import argparse
from pathlib import Path

import tiktoken
import torch
from transformers import GPT2LMHeadModel

from llm_systems_lab.config import load_experiment_config
from llm_systems_lab.models.gpt import GPT

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs"

TRANSPOSED_WEIGHTS = (
    ".attn.c_attn.weight",
    ".attn.c_proj.weight",
    ".mlp.c_fc.weight",
    ".mlp.c_proj.weight",
)


def main(args):
    config = load_experiment_config(CONFIG_DIR / args.config)
    device = args.device

    print(f"loading Hugging Face model: {args.model_name}")
    hf_model = GPT2LMHeadModel.from_pretrained(args.model_name).to(device)
    hf_model.eval()

    hf_config = hf_model.config
    config_values = (
        config.model.vocab_size,
        config.model.context_length,
        config.model.d_model,
        config.model.num_layers,
        config.model.num_heads,
    )
    hf_values = (
        hf_config.vocab_size,
        hf_config.n_positions,
        hf_config.n_embd,
        hf_config.n_layer,
        hf_config.n_head,
    )

    if config_values != hf_values:
        raise ValueError(
            f"Config does not match Hugging Face model: "
            f"config={config_values}, HF={hf_values}"
        )

    model = GPT(config.model).to(device)
    hf_state_dict = hf_model.state_dict()
    state_dict = model.state_dict()

    for key, target in state_dict.items():
        if key not in hf_state_dict:
            # The causal mask is deterministic and is not stored by newer
            # Transformers versions.
            if key.endswith(".attn.bias"):
                continue
            raise KeyError(f"Missing key in Hugging Face state_dict: {key}")

        source = hf_state_dict[key]
        if key.endswith(TRANSPOSED_WEIGHTS):
            source = source.T.contiguous()

        if source.shape != target.shape:
            raise ValueError(
                f"Shape mismatch for {key}: "
                f"HF={tuple(source.shape)}, ours={tuple(target.shape)}"
            )

        state_dict[key] = source.to(dtype=target.dtype)

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    encoding = tiktoken.get_encoding(config.data.tokenizer)
    input_ids = torch.tensor(
        [encoding.encode(args.prompt)],
        dtype=torch.long,
        device=device,
    )

    with torch.inference_mode():
        hf_logits = hf_model(input_ids).logits[:, [-1], :]
        logits, _ = model(input_ids)

    max_difference = (hf_logits - logits).abs().max().item()
    print(f"max logits difference: {max_difference:.6e}")

    if not torch.allclose(hf_logits, logits, rtol=1e-4, atol=1e-5):
        raise RuntimeError("Our model and Hugging Face GPT-2 disagree.")

    if args.output is None:
        output_path = (
            ROOT
            / config.run.output_dir
            / config.run.name
            / "checkpoints"
            / "step_000000.pt"
        )
    else:
        output_path = ROOT / args.output

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": 0,
            "model_state_dict": model.state_dict(),
            "source_model": args.model_name,
            "max_logits_difference": max_difference,
        },
        output_path,
    )

    print(f"converted checkpoint saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="gpt2_hf.toml",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="openai-community/gpt2",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Hello World!",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
    )

    main(parser.parse_args())
