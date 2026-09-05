from pathlib import Path
from urllib.request import urlopen

import tiktoken
import torch

ROOT = Path(__file__).resolve().parents[2]

RAW_PATH = ROOT / "data" / "raw" / "tinyshakespeare.txt"
PROCESSED_DIR = ROOT / "data" / "processed" / "shakespeare"

DATA_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/"
    "master/data/tinyshakespeare/input.txt"
)

def main():
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if not RAW_PATH.exists():
        with urlopen(DATA_URL) as response:
            RAW_PATH.write_bytes(response.read())

    text = RAW_PATH.read_text(encoding="utf-8")

    split_index = int(0.9 * len(text))
    train_text = text[:split_index]
    val_text = text[split_index:]

    encoding = tiktoken.get_encoding("gpt2")

    train_ids = encoding.encode(train_text)
    val_ids = encoding.encode(val_text)

    torch.save(
        torch.tensor(train_ids, dtype=torch.long),
        PROCESSED_DIR / "train.pt"
    )
    torch.save(
        torch.tensor(val_ids, dtype=torch.long),
        PROCESSED_DIR / "val.pt"
    )

    print(f"vocab_size: {encoding.n_vocab}")        # 50257
    print(f"train tokens: {len(train_ids):,}")      # 301,966
    print(f"val tokens: {len(val_ids):,}")          # 36,059

if __name__ == "__main__":
    main()