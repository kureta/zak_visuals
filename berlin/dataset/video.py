import os

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

VIDEO_DIR = '/mnt/fad02469-bb9a-4dec-a21e-8b2babc96027/datasets/berlin/rendered-video/'
VIDEO_FILE_NAME = '{}.mov'
HDF5_PATH = 'data/video_{}.hdf5'


class Video(Dataset):
    def __init__(self, video_idx, gray_scale=False, limit=None):
        super(Video, self).__init__()
        hdf5_path = HDF5_PATH.format(video_idx)
        if not os.path.exists(hdf5_path):
            size = (1024, 1024)
            video_path = os.path.join(VIDEO_DIR, VIDEO_FILE_NAME.format(video_idx + 1))
            cap = cv2.VideoCapture(video_path)
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            f = h5py.File(hdf5_path, 'w')
            dset = f.create_dataset('video', (length, *size, 3), dtype='uint8')

            idx = 0
            while True:
                result, frame = cap.read()
                if not result:
                    break
                dset[idx] = cv2.resize(frame, size, interpolation=cv2.INTER_LINEAR)
                idx += 1

            f.close()

        self.data = h5py.File(hdf5_path, 'r')['video']
        self.label = video_idx
        self.gray_scale = gray_scale
        self.limit = limit

    def __len__(self):
        if self.limit:
            return self.limit
        else:
            return self.data.shape[0]

    def __getitem__(self, idx):
        image = self.data[idx]
        if self.gray_scale:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = np.expand_dims(image, -1)
        image = image.astype('float32') / 255 * 2 - 1
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)
        return image, self.label
