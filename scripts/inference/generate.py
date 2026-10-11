import argparse
from pathlib import Path

import tiktoken
import torch

from llm_systems_lab.config import load_inference_config
from llm_systems_lab.models.gpt import GPT

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs"

def main(args):
    config = load_inference_config(CONFIG_DIR / args.config)

    device = args.device

    model = GPT(config.model).to(device)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = ROOT / checkpoint_path

    checkpoint = torch.load(
        checkpoint_path,
        map_location=torch.device(device),
    )

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    checkpoint_context_length = state_dict["transformer.wpe.weight"].shape[0]
    if checkpoint_context_length != config.model.context_length:
        raise ValueError(
            "Config/checkpoint mismatch: "
            f"config context_length={config.model.context_length}, "
            f"checkpoint context_length={checkpoint_context_length}. "
            "Use a config and checkpoint from the same experiment."
        )

    # Older checkpoints stored the deterministic causal masks alongside weights.
    # The model now recreates them for naive attention; SDPA does not need them.
    state_dict = {
        key: value for key, value in state_dict.items()
        if not key.endswith(".attn.bias")
    }
    model.load_state_dict(state_dict)

    encoding = tiktoken.get_encoding(config.tokenizer)

    prompt = args.prompt

    tokens = encoding.encode(prompt)

    if args.use_cache:
        if not tokens or args.max_new_tokens < 0:
            raise ValueError("Use a non-empty prompt and non-negative max-new-tokens")
        if len(tokens) + args.max_new_tokens > config.model.context_length:
            raise ValueError("Cached generation needs prompt + max-new-tokens within context_length")

    tokens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

    model.eval()

    with torch.inference_mode():
        cache = None
        for _ in range(args.max_new_tokens):
            if args.use_cache:
                input_ids = tokens if cache is None else tokens[:, -1:]
                logits, cache = model.forward_cached(input_ids, cache)
            else:
                input_ids = tokens[:, -config.model.context_length :]
                logits, _ = model(input_ids)

            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)

            tokens = torch.cat((tokens, next_token), dim=1)

            if next_token.item() == encoding.eot_token:
                break

    print(encoding.decode(tokens[0].tolist()))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="gpt_tiny.toml",
        help="Name of a config file inside the configs/ directory.",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="artifacts/runs/gpt-tiny-shakespeare-smoke/checkpoints/step_000500.pt",
        help="Path to a model checkpoint.",
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default="Hello there!",
        help="Message to LLM you want to send.",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=20,
        help="Maximum length (in tokens) of model responses."
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run inference on."
    )

    parser.add_argument("--use-cache", action="store_true", help="Reuse K/V; do not crop the context.")

    args = parser.parse_args()

    main(args)
