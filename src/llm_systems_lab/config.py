import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


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

@dataclass(frozen=True)
class DataConfig:

    train_path: str
    val_path: str
    tokenizer: str
@dataclass(frozen=True)
class TrainConfig:

    batch_size: int
    sequence_length: int
    micro_batch_size: int
    learning_rate: float
    max_steps: int
    seed: int
    device: str
    dtype: str
    shuffle: bool

    def __post_init__(self):
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

        if self.micro_batch_size <= 0:
            raise ValueError("micro_batch_size must be positive")
                
        if self.batch_size % self.micro_batch_size != 0:
            raise ValueError(
                "batch_size must be divisible by micro_batch_size"
            )

    @property
    def gradient_accumulation_steps(self) -> int:
        return self.batch_size // self.micro_batch_size


@dataclass(frozen=True)
class BenchmarkConfig:
    model: ModelConfig
    batch_size: int
    sequence_length: int
    warmup_iterations: int
    iterations: int
    learning_rate: float
    seed: int
    device: str
    dtype: str

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be positive")

        if self.sequence_length > self.model.context_length:
            raise ValueError("sequence_length cannot exceed context_length")

        if self.warmup_iterations < 0:
            raise ValueError("warmup_iterations cannot be negative")

        if self.iterations <= 0:
            raise ValueError("iterations must be positive")


@dataclass(frozen=True)
class EvalConfig:
    interval: int
    num_batches: int
@dataclass(frozen=True)
class CheckpointConfig:
    interval: int
@dataclass(frozen=True)
class RunConfig:
    name: str
    output_dir: str

def load_config(config_type: str, path: str | Path):
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "rb") as f:
        raw_config = tomllib.load(f)

    config = raw_config[config_type]

    config_types = {
        "model": ModelConfig,
        "data": DataConfig,
        "train": TrainConfig,
        "eval": EvalConfig,
        "checkpoint": CheckpointConfig,
        "run": RunConfig
    }

    return config_types[config_type](**config)

@dataclass(frozen=True)
class ExperimentConfig:
    model: ModelConfig
    data: DataConfig
    train: TrainConfig
    eval: EvalConfig
    checkpoint: CheckpointConfig
    run: RunConfig

    def __post_init__(self):
        if self.train.sequence_length > self.model.context_length:
            raise ValueError(
                "sequence_length cannot exceed context_length"
            )


def _load_raw_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "rb") as file:
        return tomllib.load(file)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    raw_config = _load_raw_config(path)

    return ExperimentConfig(
        model=ModelConfig(**raw_config["model"]),
        data=DataConfig(**raw_config["data"]),
        train=TrainConfig(**raw_config["train"]),
        eval=EvalConfig(**raw_config["eval"]),
        checkpoint=CheckpointConfig(**raw_config["checkpoint"]),
        run=RunConfig(**raw_config["run"]),
    )


def load_benchmark_config(path: str | Path) -> BenchmarkConfig:
    raw_config = _load_raw_config(path)

    return BenchmarkConfig(
        model=ModelConfig(**raw_config["model"]),
        **raw_config["benchmark"],
    )
