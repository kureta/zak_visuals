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
from berlin.pg_gan.utils import hypersphere
from zak_visuals.nodes.base_nodes import BaseNode

DEVICE = 'cuda:0'

vertex = """
    uniform float scale;
    attribute vec2 position;
    attribute vec2 texcoord;
    varying vec2 v_texcoord   ;
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

label_groups = [bugs, instruments, mechanical, architectural]


class AudioProcessor(BaseNode):
    def __init__(self, incoming: mp.Array, outgoing: mp.Array):
        super().__init__()
        self.incoming = incoming
        self.outgoing = outgoing

    def task(self):
        buffer = np.ndarray((2048,), dtype='float32', buffer=self.incoming.get_obj())

        stft = librosa.stft(buffer, n_fft=2048, hop_length=2048, center=False, window='boxcar')
        stft = np.abs(stft).squeeze(1).astype('float32')
        stft = 2 * stft / 2048
        stft = stft[0:128]
        self.outgoing[:] = stft[:]


class PGGAN(BaseNode):
    def __init__(self, pause_event: mp.Event, incoming: mp.Array, outgoing: mp.Queue):
        super().__init__(pause_event=pause_event)
        self.incoming = incoming
        self.outgoing = outgoing
        checkpoint_path = 'saves/zak1.1/Models/Gs_nch-4_epoch-347.pth'
        self.generator: Generator = torch.load(checkpoint_path, map_location=DEVICE)

    def task(self):
        stft = np.ndarray((128,), dtype='float32', buffer=self.incoming.get_obj())

        features = np.zeros((1, 128, 1, 1), dtype='float32')
        features[0, :, 0, 0] = stft
        features = torch.from_numpy(features)
        features = features.to(DEVICE)
        with torch.no_grad():
            image = self.generator(features)
            image = F.interpolate(image, (1920, 1080))

            image.squeeze_(0)
            image = image.permute(1, 2, 0)
            image = image.expand(1920, 1080, 3)

        self.outgoing.put(image)


class NoiseGenerator(BaseNode):
    def __init__(self, outgoing: mp.Queue, params: dict):
        super().__init__()
        self.outgoing = outgoing
        self.params = params
        self.num_frames = 128
        self.speed = 1
        self.sampling_radius = 0.01
        self.frame = self.num_frames
        self.first = True

    def setup(self):
        self.endpoints_1 = hypersphere(torch.randn(1, 128, 2, device=DEVICE), radius=self.sampling_radius)
        self.animation_1 = torch.zeros(self.num_frames, 1, 128, device=DEVICE)
        self.animation_1 = F.interpolate(self.endpoints_1, (self.num_frames,),
                                         mode='linear', align_corners=True).permute(2, 0, 1)
        self.endpoints_2 = self.endpoints_1.clone()
        self.animation_2 = self.animation_1.clone()

    def restart(self):
        self.frame = 0
        self.sampling_radius = self.params['noise_std'].value * 12. + 0.01
        if self.first:
            self.endpoints_2[:, :, 0] = self.endpoints_1[:, :, 1]
            self.endpoints_2[:, :, 1] = hypersphere(torch.randn(1, 128, device=DEVICE), radius=self.sampling_radius)
            self.animation_2[:] = F.interpolate(self.endpoints_2, (self.num_frames,),
                                                mode='linear', align_corners=True).permute(2, 0, 1)
        else:
            self.endpoints_1[:, :, 0] = self.endpoints_2[:, :, 1]
            self.endpoints_1[:, :, 1] = hypersphere(torch.randn(1, 128, device=DEVICE), radius=self.sampling_radius)
            self.animation_1[:] = F.interpolate(self.endpoints_1, (self.num_frames,),
                                                mode='linear', align_corners=True).permute(2, 0, 1)

        self.first = not self.first

    def task(self):
        self.speed = int(self.params['noise_speed'].value * 127 + 1)
        if self.params['animate_noise'].value and self.frame >= self.num_frames:
            self.restart()

        current = self.animation_1 if self.first else self.animation_2

        if self.frame >= self.num_frames:
            self.outgoing.put(current[-1])
        else:
            self.outgoing.put(current[self.frame])
            self.frame += self.speed


class LabelGenerator(BaseNode):
    def __init__(self, outgoing: mp.Queue, params: dict):
        super().__init__()
        self.outgoing = outgoing
        self.params = params
        self.num_frames = 128
        self.speed = 1
        self.frame = self.num_frames
        self.first = True

    def setup(self):
        self.label_group = label_groups[0]

        self.labels_1 = random.sample(list(self.label_group.values()), 2)
        self.endpoints_1 = torch.zeros(1, 1000, 2, device=DEVICE)
        self.endpoints_1[:, self.labels_1[0], 0] = 1.
        self.endpoints_1[:, self.labels_1[1], 1] = 1.
        self.animation_1 = torch.zeros(self.num_frames, 1, 1000, device=DEVICE)
        self.animation_1[:] = F.interpolate(self.endpoints_1, (self.num_frames,),
                                            mode='linear', align_corners=True).permute(2, 0, 1)

        self.labels_2 = self.labels_1[:]
        self.endpoints_2 = self.endpoints_1.clone()
        self.animation_2 = self.animation_1.clone()

    def restart(self):
        self.frame = 0
        category = self.params['label_group'].value
        self.label_group = label_groups[category]

        if self.first:
            self.labels_2[0] = self.labels_1[1]
            self.labels_2[1] = random.choice(list(self.label_group.values()))
            self.endpoints_2 = torch.zeros(1, 1000, 2, device=DEVICE)
            self.endpoints_2[:, self.labels_2[0], 0] = 1.
            self.endpoints_2[:, self.labels_2[1], 1] = 1.
            self.animation_2[:] = F.interpolate(self.endpoints_2, (self.num_frames,),
                                                mode='linear', align_corners=True).permute(2, 0, 1)
        else:
            self.labels_1[0] = self.labels_2[1]
            self.labels_1[1] = random.choice(list(self.label_group.values()))
            self.endpoints_1 = torch.zeros(1, 1000, 2, device=DEVICE)
            self.endpoints_1[:, self.labels_1[0], 0] = 1.
            self.endpoints_1[:, self.labels_1[1], 1] = 1.
            self.animation_1[:] = F.interpolate(self.endpoints_1, (self.num_frames,),
                                                mode='linear', align_corners=True).permute(2, 0, 1)

        self.first = not self.first

    def task(self):
        self.speed = int(self.params['label_speed'].value * 31 + 1)
        if self.params['randomize_label'].value and self.frame >= self.num_frames:
            self.restart()
            print('Label changed.')

        current = self.animation_1 if self.first else self.animation_2

        if self.frame >= self.num_frames:
            self.outgoing.put(current[-1])
        else:
            self.outgoing.put(current[self.frame])
            self.frame += self.speed


class BIGGAN(BaseNode):
    def __init__(self, stft_in: mp.Array, noise_in: mp.Queue, label_in: mp.Queue, outgoing: mp.Queue, params: dict,
                 pause_event: mp.Event):
        super().__init__(pause_event=pause_event)
        self.stft_in = stft_in
        self.noise_in = noise_in
        self.label_in = label_in
        self.outgoing = outgoing
        self.params = params

        self.generator = BigGAN.from_pretrained('biggan-deep-512')
        self.generator.to(DEVICE)

    def task(self):
        noise = self.noise_in.get()
        label = self.label_in.get()
        stft = np.ndarray((128,), dtype='float32', buffer=self.stft_in.get_obj())

        scale = self.params['stft_scale'].value * 250

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

        self.outgoing.put(image)


class ImageFX(BaseNode):
    def __init__(self, incoming: mp.Queue, outgoing: mp.Queue, params: dict):
        super().__init__()
        self.incoming = incoming
        self.outgoing = outgoing
        self.params = params

    def task(self):
        image: torch.Tensor = self.incoming.get()

        rgb = self.params['rgb'].value * 50

        cimage = cp.asarray(image)
        for idx in range(3):
            shift = np.random.randn(2).astype('float32') * rgb
            cimage[:, :, idx] = cpi.shift(cimage[:, :, idx], shift)

        cimage = (cimage + 1.) / 2 * 255
        cimage = cp.clip(cimage, 0, 255).astype(cp.uint8)
        nimage = cp.asnumpy(cimage)

        self.outgoing.put(nimage)


class InteropDisplay(BaseNode):
    def __init__(self, incoming: mp.Queue, exit_app: mp.Event):
        super().__init__()
        self.incoming = incoming
        self.exit_app = exit_app

    def setup(self):
        from glumpy import app
        self.window = app.Window(width=1280, height=720, fullscreen=False, decoration=True)

        self.quad = gloo.Program(vertex, fragment, count=4)
        self.quad['position'] = [(-1, -1), (-1, +1), (+1, -1), (+1, +1)]
        self.quad['texcoord'] = [(0, 1), (0, 0), (1, 1), (1, 0)]
        self.quad['texture'] = np.zeros((1920, 1080, 3), dtype='uint8')

        self.backend = app.__backend__
        self.clock = app.__init__(backend=self.backend)

    def task(self):
        image = self.incoming.get()

        self.window.set_title(f'{self.window.fps:.2f}')
        self.quad['texture'] = image
        self.window.clear()
        self.quad.draw(gl.GL_TRIANGLE_STRIP)

        if not self.backend.process(self.clock.tick()):
            self.exit_app.set()
