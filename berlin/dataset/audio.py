import os

import essentia.standard as es
import numpy as np
import torch
from numpy.lib.stride_tricks import as_strided
from torch.utils.data import Dataset

VIDEO_DIR = '/home/kureta/Videos/Rendered/'
AUDIO_FILE_NAME = '{}.wav'


class Audio(Dataset):
    def __init__(self, video_idx, limit=None):
        super(Audio, self).__init__()

        audio_path = os.path.join(VIDEO_DIR, AUDIO_FILE_NAME.format(video_idx + 1))

        self.label = video_idx
        easy_loader = es.EasyLoader(filename=audio_path,
                                    replayGain=0,
                                    sampleRate=48000)

        self.window = es.Windowing(size=2048)
        self.spectrum = es.Spectrum(size=2048)
        self.mel = es.MelBands(inputSize=1025, highFrequencyBound=12000, lowFrequencyBound=46.875, numberBands=32,
                               sampleRate=48000)

        self.audio = easy_loader()
        self.length = len(self.audio) // (48000 // 25)
        self.audio = np.pad(self.audio, 64, 'reflect')
        self.frames = as_strided(self.audio,
                                 (self.length, 2048),
                                 ((48000 // 25) * self.audio.strides[0], self.audio.strides[0]),
                                 writeable=False)
        if limit:
            self.length = limit

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        value = self.frames[idx]
        value = self.window(value)
        value = self.spectrum(value)
        value = self.mel(value)
        value = torch.from_numpy(value)

        return value, self.label
