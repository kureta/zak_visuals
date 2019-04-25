from torch.utils.data import Dataset

from berlin.dataset.audio import Audio
from berlin.dataset.video import Video


class ConditionalDataset(Dataset):
    def __init__(self, video_idx, gray_scale=False, limit=None):
        super(ConditionalDataset, self).__init__()

        self.label = video_idx
        self.image_data = Video(video_idx, gray_scale=gray_scale, limit=limit)
        self.audio_data = Audio(video_idx, limit=limit)

    def __len__(self):
        len_img = len(self.image_data)
        len_audio = len(self.audio_data)

        if len_img != len_audio:
            raise ValueError('Image and audio data are not the same size. Probably misaligned audio.')

        return len_img

    def __getitem__(self, idx):
        image, _ = self.image_data[idx]
        mel, _ = self.audio_data[idx]
        return image, mel, self.label
