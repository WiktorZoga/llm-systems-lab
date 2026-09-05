from torch.utils.data import DataLoader

class InfiniteDataLoader:
    def __init__(self, dataloader: DataLoader):
        self.dataloader = dataloader

    def __iter__(self):
        while True:
            yield from self.dataloader