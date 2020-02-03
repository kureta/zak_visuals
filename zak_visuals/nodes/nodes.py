import random

import cupy as cp
import cupyx.scipy.ndimage as cpi
import librosa
import numpy as np
import torch
from glumpy import gloo, gl
from pytorch_pretrained_biggan import BigGAN
from torch import multiprocessing as mp
from torch.nn import functional as F

from berlin.pg_gan.model import Generator
from zak_visuals.nodes.base_nodes import BaseNode, Edge

CHECKPOINT_PATH = 'saves/zak1.1/Models/Gs_nch-4_epoch-347.pth'
DEVICE = 'cuda:0'

vertex = """
    uniform float scale;
    attribute vec2 position;
    attribute vec2 texcoord;
    varying vec2 v_texcoord;
    void main()
    {
        gl_Position = vec4(position, 0.0, 1.0);
        v_texcoord = texcoord;
    }
"""

fragment = """
    uniform sampler2D texture;
    varying vec2 v_texcoord;
    void main()
    {
        gl_FragColor = texture2D(texture, v_texcoord);
    }
"""
bugs = {
    'fly': 308,
    'ant': 310,
    'roach': 314,
    'mantis': 315,
    'cicada': 316,
    'web': 815,
}

instruments = {
    'accordion': 401,
    'basson': 432,
    'cello': 486,
    'trumpte': 513,
    'drum': 541,
    'flute': 558,
    'horn': 566,
    'grand piano': 579,
    'harp': 594,
    'trombone': 875,
    'upright piano': 881,
    'violin': 889,
}

mechanical = {
    'clock': 409,
    'barometer': 426,
    'dial phone': 528,
    'disk brake': 535,
    'compass': 635,
}

architectural = {
    'altar': 406,
    'lighthouse': 437,
    'birdhouse': 448,
    'church': 497,
    'dome': 538,
    'mosque': 668,
    'prison': 743,
}


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

    def cleanup(self):
        self.outgoing.cleanup_output()


class PGGAN(BaseNode):
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

    def cleanup(self):
        self.outgoing.cleanup_output()
        self.incoming.cleanup_input()


class NoiseGenerator(BaseNode):
    def __init__(self, outgoing: Edge):
        super().__init__()
        self.outgoing = outgoing

    def setup(self):
        self.num_frames = 32
        self.frame = 0
        self.noise_vector_endpoints = torch.randn(1, 128, 2, device=DEVICE) * 0.7
        self.noise_vector = F.interpolate(self.noise_vector_endpoints,
                                          (self.num_frames,), mode='linear', align_corners=True).permute(2, 0, 1)

    def run(self):
        if self.frame >= len(self.noise_vector):
            self.outgoing.write(self.noise_vector[-1])
        else:
            self.outgoing.write(self.noise_vector[self.frame])
            self.frame += 1

    def cleanup(self):
        self.outgoing.cleanup_output()


class LabelGenerator(BaseNode):
    def __init__(self, outgoing: Edge):
        super().__init__()
        self.outgoing = outgoing

    def setup(self):
        num_frames = 32
        self.frame = 0
        self.labels = random.sample(list(bugs.values()), 2)
        self.label_endpoints = torch.zeros(1, 1000, 2, device=DEVICE)
        self.label_endpoints[:, self.labels[0], 0] = 1.
        self.label_endpoints[:, self.labels[1], 1] = 1.
        self.label_vector = F.interpolate(self.label_endpoints,
                                          (num_frames,), mode='linear', align_corners=True).permute(2, 0, 1)

    def run(self):
        if self.frame >= len(self.label_vector):
            self.outgoing.write(self.label_vector[-1])
        else:
            self.outgoing.write(self.label_vector[self.frame])
            self.frame += 1

    def cleanup(self):
        self.outgoing.cleanup_output()


class BIGGAN(BaseNode):
    def __init__(self, stft_in: Edge, noise_in: Edge, label_in: Edge, outgoing: Edge, noise_scale: mp.Value):
        super().__init__()
        self.stft_in = stft_in
        self.noise_in = noise_in
        self.label_in = label_in
        self.outgoing = outgoing
        self.noise_scale = noise_scale

        self.generator = BigGAN.from_pretrained('biggan-deep-512')
        self.generator.to(DEVICE)

    def run(self):
        stft = self.stft_in.read()
        noise = self.noise_in.read()
        label = self.label_in.read()
        for element in [stft, noise, label]:
            if element is None:
                return

        scale = self.noise_scale.value * 250

        features = np.zeros((1, 128), dtype='float32')
        features[0, :] = stft * scale

        features = torch.from_numpy(features)
        features = features.to(DEVICE)
        features = features + noise

        with torch.no_grad():
            image = self.generator(features, label, 0.4)
            image = torch.nn.functional.interpolate(image, (1920, 1080), mode='bicubic')

            image.squeeze_(0)
            image = image.permute(1, 2, 0)

        self.outgoing.write(image)

    def cleanup(self):
        self.outgoing.cleanup_output()
        self.stft_in.cleanup_input()
        self.noise_in.cleanup_input()
        self.label_in.cleanup_input()


class ImageFX(BaseNode):
    def __init__(self, incoming: Edge, outgoing: Edge, rgb_intensity: mp.Value):
        super().__init__()
        self.incoming = incoming
        self.outgoing = outgoing
        self.rgb_intensity = rgb_intensity

    def run(self):
        image: torch.Tensor = self.incoming.read()
        if image is None:
            return
        rgb = self.rgb_intensity.value * 50

        cimage = cp.asarray(image)
        for idx in range(3):
            shift = np.random.randn(2).astype('float32') * rgb
            cimage[:, :, idx] = cpi.shift(cimage[:, :, idx], shift)

        cimage = (cimage + 1.) / 2 * 255
        cimage = cp.clip(cimage, 0, 255).astype(cp.uint8)
        nimage = cp.asnumpy(cimage)

        self.outgoing.write(nimage)

    def cleanup(self):
        self.outgoing.cleanup_output()
        self.incoming.cleanup_input()


class InteropDisplay(BaseNode):
    def __init__(self, incoming: Edge, exit_event: mp.Event):
        super().__init__()
        self.incoming = incoming
        self.exit_event = exit_event

    def setup(self):
        from glumpy import app
        self.window = app.Window(width=1280, height=720, fullscreen=False, decoration=True)

        self.quad = gloo.Program(vertex, fragment, count=4)
        self.quad['position'] = [(-1, -1), (-1, +1), (+1, -1), (+1, +1)]
        self.quad['texcoord'] = [(0, 1), (0, 0), (1, 1), (1, 0)]
        self.quad['texture'] = np.zeros((1920, 1080, 3), dtype='uint8')

        self.backend = app.__backend__
        self.clock = app.__init__(backend=self.backend)

    def run(self):
        image = self.incoming.read()
        if image is None:
            return

        self.window.set_title(f'{self.window.fps:.2f}')
        self.quad['texture'] = image
        self.window.clear()
        self.quad.draw(gl.GL_TRIANGLE_STRIP)

        if not self.backend.process(self.clock.tick()):
            self.exit_event.set()

    def cleanup(self):
        self.incoming.cleanup_input()
