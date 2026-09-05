import argparse
from pathlib import Path

import tiktoken
import torch

from llm_systems_lab.config import load_experiment_config
from llm_systems_lab.models.gpt import GPT

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs"

def main(args):
    config = load_experiment_config(CONFIG_DIR / args.config)

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

    model.load_state_dict(state_dict)

    encoding = tiktoken.get_encoding(config.data.tokenizer)

    prompt = args.prompt

    tokens = encoding.encode(prompt)

    tokens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

    model.eval()

    with torch.inference_mode():
        for _ in range(args.max_new_tokens):
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
        help="Maximum legnth (in tokens) of model responses."
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device you want to run inferce on."
    )

    args = parser.parse_args()

    main(args)
