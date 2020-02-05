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
    'trumpet': 513,
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
        self.count = 0
        self.mean = 0
        self.std = 0
        self.epsilon = 1e-9

    def task(self):
        buffer = np.ndarray((2048,), dtype='float32', buffer=self.incoming)

        stft = librosa.stft(buffer, n_fft=2048, hop_length=2048, center=False, window='boxcar')
        stft = np.abs(stft).squeeze(1).astype('float32')
        # stft = 2 * stft / 2048
        # stft = stft**2
        for idx in range(1, 128):
            stft[idx] = stft[idx * 8:(idx + 1) * 8].sum()

        new_count = self.count + 128
        new_mean = (self.mean * self.count + stft[:128].sum()) / new_count
        diff = stft[:128] - new_mean
        new_std = np.sqrt((np.square(self.std) * self.count + np.dot(diff, diff)) / new_count)

        self.mean = new_mean
        self.std = new_std
        self.count = new_count

        self.outgoing[:] = (stft[:128] - self.mean) / max(self.epsilon, self.std)


class PGGAN(BaseNode):
    def __init__(self, pause_event: mp.Event, incoming: mp.Array, noise: mp.Queue, outgoing: mp.Queue,
                 params: dict):
        super().__init__(pause_event=pause_event)
        self.incoming = incoming
        self.noise = noise
        self.outgoing = outgoing
        self.params = params
        checkpoint_path = 'saves/zak1.1/Models/Gs_nch-4_epoch-347.pth'
        self.generator: Generator = torch.load(checkpoint_path, map_location=DEVICE)

    def setup(self):
        torch.autograd.set_grad_enabled(False)

    def task(self):
        noise = self.noise.get().unsqueeze(2).unsqueeze(3)
        stft = np.ndarray((128,), dtype='float32', buffer=self.incoming)

        scale = self.params['stft_scale'].value * 12

        features = np.zeros((1, 128, 1, 1), dtype='float32')
        features[0, :, 0, 0] = stft
        features = torch.from_numpy(features)
        features = features.to(DEVICE)
        features = hypersphere(features, scale)
        features = features + noise

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
        self.frame = 0
        self.idx = 0
        self.moving = False

    def swap(self):
        self.buffers[:] = self.buffers[::-1]

    def set_start(self, motion):
        motion[0, :, :] = hypersphere(torch.randn(1, 128, device=DEVICE), radius=self.sampling_radius)

    def create_motion(self, motion):
        motion[-1, :, :] = hypersphere(torch.randn(1, 128, device=DEVICE), radius=self.sampling_radius)
        motion[:] = F.interpolate(torch.stack((motion[0], motion[-1])).permute(1, 2, 0),
                                  self.num_frames, mode='linear', align_corners=True).permute(2, 0, 1)

    def setup(self):
        torch.autograd.set_grad_enabled(False)
        self.motion_1 = torch.zeros(self.num_frames, 1, 128, device=DEVICE)
        self.motion_2 = torch.zeros(self.num_frames, 1, 128, device=DEVICE)
        self.buffers = [self.motion_1, self.motion_2]
        self.set_start(self.buffers[0])

    def restart(self):
        self.sampling_radius = self.params['noise_std'].value * 12. + 0.01
        self.moving = True
        self.create_motion(self.buffers[0])
        self.set_start(self.buffers[1])

    def task(self):
        self.speed = int(self.params['noise_speed'].value * 31 + 1)

        if self.frame >= self.num_frames:
            self.moving = False
            self.swap()
            self.frame = 0

        if self.params['animate_noise'].value and not self.moving:
            self.restart()

        if self.moving:
            self.outgoing.put(self.buffers[0][self.frame])
            self.frame += self.speed
        else:
            self.outgoing.put(self.buffers[0][0])


class LabelGenerator(BaseNode):
    def __init__(self, outgoing: mp.Queue, params: dict):
        super().__init__()
        self.outgoing = outgoing
        self.params = params
        self.num_frames = 128
        self.speed = 1
        self.frame = 0
        self.idx = 0
        self.moving = False

    def swap(self):
        self.buffers[:] = self.buffers[::-1]

    def get_label(self):
        category = self.params['label_group'].value
        label_group = label_groups[category]
        idx = self.idx % len(label_group)
        name, label = list(label_group.items())[idx]
        return name, label

    def set_start(self, motion):
        name, label = self.get_label()
        print(f'label: {name}')
        motion[0, :, :] = torch.zeros(1, 1000, device=DEVICE)
        motion[0, :, label] = 1.

    def create_motion(self, motion):
        name, label = self.get_label()
        motion[-1, :, :] = torch.zeros(1, 1000, device=DEVICE)
        motion[-1, :, label] = 1.
        motion[:] = F.interpolate(torch.stack((motion[0], motion[-1])).permute(1, 2, 0),
                                  self.num_frames, mode='linear', align_corners=True).permute(2, 0, 1)

    def setup(self):
        torch.autograd.set_grad_enabled(False)
        self.motion_1 = torch.zeros(self.num_frames, 1, 1000, device=DEVICE)
        self.motion_2 = torch.zeros(self.num_frames, 1, 1000, device=DEVICE)
        self.buffers = [self.motion_1, self.motion_2]
        self.set_start(self.buffers[0])

    def restart(self):
        self.idx += 1
        self.moving = True
        self.create_motion(self.buffers[0])
        self.set_start(self.buffers[1])

    def task(self):
        self.speed = int(self.params['label_speed'].value * 31 + 1)

        if self.frame >= self.num_frames:
            self.moving = False
            self.swap()
            self.frame = 0

        if self.params['randomize_label'].value and not self.moving:
            self.restart()

        if self.moving:
            self.outgoing.put(self.buffers[0][self.frame])
            self.frame += self.speed
        else:
            self.outgoing.put(self.buffers[0][0])


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

    def setup(self):
        torch.autograd.set_grad_enabled(False)

    def task(self):
        noise = self.noise_in.get()
        label = self.label_in.get()
        stft = np.ndarray((128,), dtype='float32', buffer=self.stft_in)

        scale = self.params['stft_scale'].value * 12

        features = np.zeros((1, 128), dtype='float32')
        features[0, :] = stft

        features = torch.from_numpy(features)
        features = features.to(DEVICE)
        features = hypersphere(features, scale)
        features = features + noise

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

    def setup(self):
        torch.autograd.set_grad_enabled(False)

    def task(self):
        image: torch.Tensor = self.incoming.get().clone()

        rgb = self.params['rgb'].value * 50

        cimage = cp.asarray(image)
        for idx in range(3):
            shift = np.random.randn(2).astype('float32') * rgb
            cimage[:, :, idx] = cpi.shift(cimage[:, :, idx], shift, mode='mirror')

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
