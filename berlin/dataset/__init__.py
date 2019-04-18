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


class AutoencoderDataset(Dataset):
    def __init__(self, config):
        super(AutoencoderDataset, self).__init__()

        self.config = config
        self.image_data = BaseDataset(config.image_data_path)
        self.audio_data = BaseDataset(config.audio_data_path)

    def __len__(self):
        len_img = self.image_data.data.shape[0]
        len_audio = self.audio_data.data.shape[0]

        if len_img != len_audio:
            raise ValueError('Image and audio data are not the same size. Probably misaligned audio.')

        return len_img

    def __getitem__(self, index):
        return self.image_data.data[index], self.audio_data.data[index]
