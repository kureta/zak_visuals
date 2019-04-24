import os

import cv2
import torch
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
        self.cap = cv2.VideoCapture(video_path)
        self.length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.size = size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        result, frame = self.cap.read()

        if not result:
            raise IndexError

        frame = cv2.resize(frame, self.size, interpolation=cv2.INTER_LINEAR)
        frame = torch.from_numpy((frame / 255 * 2 - 1).astype('float32'))
        frame = frame.permute((2, 0, 1))
        # frame = interpolate(frame.unsqueeze(0), self.size, mode='bilinear').squeeze(0)

        return frame, self.label
