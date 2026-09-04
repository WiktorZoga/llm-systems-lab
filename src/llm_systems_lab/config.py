import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True) 
class ModelConfig: 
    """
        Simple MHA Transformer architecture
    """
    vocab_size: int
    context_length: int
    d_model: int
    num_layers: int
    num_heads: int
    dropout: float

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")

        if self.context_length <= 0:
            raise ValueError("context_length must be positive")

        if self.d_model <= 0:
            raise ValueError("d_model must be positive")

        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")

        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")

        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in the range [0, 1)")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.num_heads

def load_model_config(path: str | Path) -> ModelConfig:
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "rb") as f:
        raw_config = tomllib.load(f)

    model_values = raw_config["model"]

    return ModelConfig(**model_values)