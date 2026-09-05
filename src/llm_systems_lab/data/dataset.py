import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, token_ids: torch.Tensor, sequence_length: int):
        if token_ids.ndim != 1:
            raise ValueError("token_ids must be a 1D tensor")

        if token_ids.dtype != torch.long:
            raise TypeError("token_ids must have dtype torch.long")

        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")

        if token_ids.numel() <= sequence_length:
            raise ValueError(
                "token_ids must contain at least sequence_length + 1 tokens"
            )

        self.token_ids = token_ids
        self.sequence_length = sequence_length

    def __len__(self):
        return self.token_ids.numel() - self.sequence_length

    def __getitem__(self, index: int):
        if index < 0 or index >= len(self):
            raise IndexError(f"dataset index out of range: {index}")

        window = self.token_ids[index : index + self.sequence_length + 1]

        input_ids = window[:-1]
        targets = window[1:]

        return input_ids, targets
