import torch


def select_device(name):
    device = torch.device(name)
    if device.type not in ("cpu", "mps", "cuda"):
        raise ValueError("Use cpu, mps, cuda or cuda:N")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available in this PyTorch process")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available in this PyTorch process")
        # Validates the index and makes unqualified CUDA operations use this GPU.
        torch.cuda.set_device(device)
        device = torch.device("cuda", torch.cuda.current_device())
    return device


def synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)
