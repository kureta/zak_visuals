import os
from itertools import islice

import torch
from skvideo.io import FFmpegReader
from torch.nn.functional import interpolate
from torch.utils.data import Dataset

VIDEO_DIR = '/home/kureta/Videos/Rendered/'
VIDEO_FILE_NAME = '{}.mov'
AUDIO_FILE_NAME = '{}.wav'
NUM_VIDEOS = 9


class Video(Dataset):
    def __init__(self, video_idx, size=(1920, 1080)):
        super(Video, self).__init__()

        video_path = os.path.join(VIDEO_DIR, VIDEO_FILE_NAME.format(video_idx + 1))

        self.label = video_idx
        self.cap = FFmpegReader(video_path)
        self.length, _, _, _ = self.cap.getShape()
        self.size = size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        video_gen = self.cap.nextFrame()

        frame = next(islice(video_gen, idx))

        frame = torch.from_numpy(((frame / 255) * 2 - 1).astype('float32'))
        frame = frame.permute((2, 0, 1))
        frame = interpolate(frame.unsqueeze(0), self.size, mode='bilinear').squeeze(0)

        return frame, self.label
