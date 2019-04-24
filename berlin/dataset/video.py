import os

import cv2
import essentia.standard as es
import torch
from torch.utils.data import Dataset

VIDEO_DIR = '/home/kureta/Videos/Rendered/'
VIDEO_FILE_NAME = '{}.mov'
AUDIO_FILE_NAME = '{}.wav'
NUM_VIDEOS = 9


class Video(Dataset):
    def __init__(self, video_idx):
        super(Video, self).__init__()

        video_path = os.path.join(VIDEO_DIR, VIDEO_FILE_NAME.format(video_idx + 1))

        self.label = video_idx
        self.cap = cv2.VideoCapture(video_path)
        self.length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.__downsample_factor = 1

    @property
    def downsample_factor(self):
        return self.__downsample_factor

    @downsample_factor.setter
    def downsample_factor(self, value):
        if 120 % value != 0:
            raise ValueError

        self.__downsample_factor = value

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        result, frame = self.cap.read()
        if not result:
            raise IndexError

        image = cv2.resize(frame, (1024, 1024), interpolation=cv2.INTER_LINEAR)
        image = torch.from_numpy((image / 255.).astype('float32'))

        return image, self.label


class Audio(Dataset):
    def __init__(self, video_idx):
        super(Audio, self).__init__()

        audio_path = os.path.join(VIDEO_DIR, AUDIO_FILE_NAME.format(video_idx + 1))

        self.label = video_idx
        easy_loader = es.EasyLoader(filename=audio_path,
                                    replayGain=0,
                                    sampleRate=48000)
        self.audio = easy_loader()
        self.length = len(self.audio)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        value = self.audio[idx]

        return value, self.label
