from torch.utils.data import Dataset
import torch
from berlin.preprocess.images import load_saved
from berlin.config import Config


class ImageDataset(Dataset):
    def __init__(self, config=Config()):
        super(Dataset, self).__init__()
        self.config = config

        images = load_saved()

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError
