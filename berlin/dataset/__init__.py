import torch
from torch.utils.data import Dataset

from berlin.config import Config


class BaseDataset(Dataset):
    def __init__(self, path, config=Config()):
        super(BaseDataset, self).__init__()

        self.config = config
        self.data = torch.load(path)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index]
