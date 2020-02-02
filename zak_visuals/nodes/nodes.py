import cv2
import librosa
import numpy as np
import torch
from pytorch_pretrained_biggan import BigGAN, one_hot_from_names
from torch import multiprocessing as mp
from torch.nn import functional as F

from berlin.pg_gan.model import Generator
from zak_visuals.nodes.base_nodes import BaseNode, Edge

CHECKPOINT_PATH = 'saves/zak1.1/Models/Gs_nch-4_epoch-347.pth'
DEVICE = 'cuda:0'


class AudioProcessor(BaseNode):
    def __init__(self, incoming: mp.Array, outgoing: Edge):
        super().__init__()
        self.incoming = incoming
        self.outgoing = outgoing

    def run(self):
        buffer = np.ndarray((2048,), dtype='float32', buffer=self.incoming.get_obj())

        stft = librosa.stft(buffer, n_fft=2048, hop_length=2048, center=False, window='boxcar')
        stft = np.abs(stft).squeeze(1).astype('float32')
        stft = 2 * stft / 2048
        stft = stft[0:128]
        self.outgoing.write(stft)


class ImageGenerator(BaseNode):
    def __init__(self, incoming: Edge, outgoing: Edge):
        super().__init__()
        self.incoming = incoming
        self.outgoing = outgoing
        self.generator: Generator = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

    def run(self):
        stft = self.incoming.read()
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

        self.outgoing.write(image)


class NoiseGenerator(BaseNode):
    def __init__(self, outgoing: Edge):
        super().__init__()
        self.outgoing = outgoing

        self.num_frames = 32

    def setup(self):
        self.noise_vector_endpoints = torch.randn(1, 128, 2, device=DEVICE) * 0.7
        self.noise_vector = F.interpolate(self.noise_vector_endpoints,
                                          (self.num_frames,), mode='linear', align_corners=True).permute(2, 0, 1)

    def run(self):
        self.noise_vector_endpoints[:, :, 0] = self.noise_vector_endpoints[:, :, 1]
        self.noise_vector_endpoints[:, :, 1].normal_(std=0.7)
        self.noise_vector = F.interpolate(self.noise_vector_endpoints,
                                          (self.num_frames,), mode='linear', align_corners=True).permute(2, 0, 1)

        self.outgoing.write(self.noise_vector)


class AlternativeGenerator(BaseNode):
    def __init__(self, stft_in: Edge, noise_in: Edge, outgoing: Edge, noise_scale: mp.Value):
        super().__init__()
        self.stft_in = stft_in
        self.noise_in = noise_in
        self.outgoing = outgoing
        self.noise_scale = noise_scale

        self.generator = BigGAN.from_pretrained('biggan-deep-256')
        self.generator.to(DEVICE)

        labels = ['analog clock']
        class_vector = one_hot_from_names(labels)
        self.class_vector = torch.from_numpy(class_vector).to(DEVICE)

        self.frame = 0

        self.current_motion = None

    def run(self):
        stft = self.stft_in.read()
        if stft is None:
            return
        if self.frame == 0:
            self.current_motion = self.noise_in.read()
        if self.current_motion is None:
            return

        scale = self.noise_scale.value

        features = np.zeros((1, 128), dtype='float32')
        features[0, :] = stft * scale

        features = torch.from_numpy(features)
        features = features.to(DEVICE)
        features = features + self.current_motion[self.frame]
        self.frame = (self.frame + 1) % self.current_motion.shape[0]

        with torch.no_grad():
            image = self.generator(features, self.class_vector, 0.4)
            image = torch.nn.functional.interpolate(image, (1920, 1080), mode='bicubic')

            image = (image + 1) / 2
            image = image * 255
            image.squeeze_(0)
            image = image.permute(1, 2, 0)

            image = image.cpu().numpy().astype('uint8')

        self.outgoing.write(image)


class ImageFX(BaseNode):
    def __init__(self, incoming: Edge, outgoing: Edge, rgb_intensity: mp.Value):
        super().__init__()
        self.incoming = incoming
        self.outgoing = outgoing
        self.rgb_intensity = rgb_intensity

    def run(self):
        image = self.incoming.read()
        if image is None:
            return

        base_points = np.float32([[420, 1080], [420, 0], [1500, 0]])
        rgb = self.rgb_intensity.value
        for idx in range(3):
            shift = np.random.normal(0, rgb, (3, 2)).astype('float32')
            m = cv2.getAffineTransform(base_points, base_points + shift)
            image[:, :, idx] = cv2.warpAffine(image[:, :, idx], m, (1080, 1920))

        self.outgoing.write(image)


class ImageDisplay(BaseNode):
    def __init__(self, incoming: Edge, exit_event: mp.Event):
        super().__init__()
        self.incoming = incoming
        self.exit_event = exit_event

    def setup(self):
        cv2.namedWindow('frame', cv2.WINDOW_FREERATIO)
        cv2.setWindowProperty('frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    def run(self):
        image = self.incoming.read()
        if image is None:
            return

        cv2.imshow('frame', image)
        if cv2.waitKey(1) == ord('q'):
            self.exit_event.set()
