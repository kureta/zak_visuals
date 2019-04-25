import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

VIDEO_DIR = '/home/kureta/Videos/Rendered/'
VIDEO_FILE_NAME = '{}.mov'
AUDIO_FILE_NAME = '{}.wav'
NUM_VIDEOS = 9


class Video(Dataset):
    def __init__(self, video_idx, size=(1920, 1080), num_frames=200):
        super(Video, self).__init__()

        video_path = os.path.join(VIDEO_DIR, VIDEO_FILE_NAME.format(video_idx + 1))

        self.cap = cv2.VideoCapture(video_path)
        length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.label = video_idx

        indices = np.random.permutation(length)[:num_frames]

        frames = np.empty((num_frames, *size), dtype='float32')

        for i, idx in enumerate(indices):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            result, frame = self.cap.read()
            if not result:
                raise IndexError
            grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            del frame
            frames[i] = cv2.resize(grayscale, size, interpolation=cv2.INTER_LINEAR).astype('float32')
            del grayscale

        frames_tensor = torch.from_numpy(frames)
        del frames

        frames_tensor = frames_tensor / 255 * 2 - 1
        frames_tensor.unsqueeze_(1)
        self.frames = frames_tensor
        self.length = num_frames

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.frames[idx], self.label
