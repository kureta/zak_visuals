from multiprocessing import managers

import cv2
import librosa
import numpy as np
import torch
from pytorch_pretrained_biggan import BigGAN, one_hot_from_names
from torch import multiprocessing as mp
from torch.nn import functional as F

from zak_visuals.nodes.base_nodes import ProcessorNode, OutputNode
from zak_visuals.pg_gan.model import Generator

CHECKPOINT_PATH = 'saves/zak1.1/Models/Gs_nch-4_epoch-347.pth'
DEVICE = 'cuda:0'


class AudioProcessor(ProcessorNode):
    def run(self):
        buffer = self.incoming
        if buffer is None:
            return
        stft = librosa.stft(buffer, n_fft=2048, hop_length=2048, center=False, window='boxcar')
        stft = np.abs(stft).squeeze(1).astype('float32')
        stft = 2 * stft / 2048
        # stft = resample(stft, 128)
        stft = stft[0:128]
        self.outgoing = stft


class ImageGenerator(ProcessorNode):
    def setup(self):
        self.generator: Generator = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

    def run(self):
        stft = self.incoming
        if stft is None:
            return
        features = np.zeros((1, 128, 1, 1), dtype='float32')
        features[0, :, 0, 0] = stft
        features = torch.from_numpy(features)
        features = features.to(DEVICE)
        with torch.no_grad():
            image = self.generator(features)
            image = F.interpolate(image, (1920, 1080))

            image = (image + 1) / 2
            image = image * 255
            image.squeeze_(0)
            image = image.permute(1, 2, 0)
            image = image.expand(1920, 1080, 3)

            image = image.cpu().numpy().astype('uint8')

        self.outgoing = image


class AlternativeGenerator(ProcessorNode):
    def __init__(self, incoming: mp.Queue, outgoing: mp.Queue, osc_params: managers.Namespace):
        super().__init__(incoming, outgoing)
        self.osc_params = osc_params

    def setup(self):
        self.generator = BigGAN.from_pretrained('biggan-deep-256')

        labels = ['analog clock']
        class_vector = one_hot_from_names(labels)
        self.class_vector = torch.from_numpy(class_vector).to(DEVICE)

        self.idx = 0
        self.num_frames = 16

        self.noise_vector_endpoints = torch.randn(1, 128, 2, device=DEVICE)
        self.noise_vector = F.interpolate(self.noise_vector_endpoints, (self.num_frames,), mode='linear')
        self.noise_vector.transpose_(2, 0)
        self.noise_vector.transpose_(1, 2)

        self.generator.to(DEVICE)

    def run(self):
        stft = self.incoming
        if stft is None:
            return
        try:
            scale = self.osc_params.scale
        except (BrokenPipeError, ConnectionResetError):
            return
        features = np.zeros((1, 128), dtype='float32')
        features[0, :] = stft * scale

        features = torch.from_numpy(features)
        features = features.to(DEVICE)
        features = features + self.noise_vector[self.idx]
        self.idx = (self.idx + 1) % self.num_frames
        if self.idx == 0:
            self.noise_vector_endpoints[:, :, 0] = self.noise_vector_endpoints[:, :, 1]
            self.noise_vector_endpoints[:, :, 1].normal_()
            self.noise_vector = F.interpolate(self.noise_vector_endpoints, (self.num_frames,), mode='linear')
            self.noise_vector.transpose_(2, 0)
            self.noise_vector.transpose_(1, 2)
        with torch.no_grad():
            image = self.generator(features, self.class_vector, 0.4)
            image = torch.nn.functional.interpolate(image, (1920, 1080), mode='bicubic')

            image = (image + 1) / 2
            image = image * 255
            image.squeeze_(0)
            image = image.permute(1, 2, 0)

            image = image.cpu().numpy().astype('uint8')

        self.outgoing = image


class ImageFX(ProcessorNode):
    def __init__(self, incoming: mp.Queue, outgoing: mp.Queue, osc_params: managers.Namespace):
        super().__init__(incoming, outgoing)
        self.osc_params = osc_params

    def run(self):
        image = self.incoming
        if image is None:
            return

        base_points = np.float32([[420, 1080], [420, 0], [1500, 0]])
        try:
            rgb = self.osc_params.rgb_intensity
        except (BrokenPipeError, ConnectionResetError):
            return
        for idx in range(3):
            shift = np.random.normal(0, rgb, (3, 2)).astype('float32')
            m = cv2.getAffineTransform(base_points, base_points + shift)
            image[:, :, idx] = cv2.warpAffine(image[:, :, idx], m, (1080, 1920))

        self.outgoing = image


class ImageDisplay(OutputNode):
    def __init__(self, incoming, exit_event):
        super().__init__(incoming=incoming)
        self.exit_event = exit_event

    def setup(self):
        cv2.namedWindow('frame', cv2.WINDOW_FREERATIO)
        cv2.setWindowProperty('frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    def run(self):
        image = self.incoming
        if image is None:
            return

        cv2.imshow('frame', image)
        if cv2.waitKey(1) == ord('q'):
            self.exit_event.set()
