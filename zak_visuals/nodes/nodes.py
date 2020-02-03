from contextlib import contextmanager

import cupy as cp
import cupyx.scipy.ndimage as cpi
import cv2
import librosa
import numpy as np
import pycuda
import pycuda.driver
import torch
from glumpy import app, gloo, gl
from pycuda.gl import graphics_map_flags
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

    def cleanup(self):
        self.outgoing.cleanup_output()


class BIGGAN(BaseNode):
    def __init__(self, stft_in: Edge, noise_in: Edge, outgoing: Edge, noise_scale: mp.Value):
        super().__init__()
        self.stft_in = stft_in
        self.noise_in = noise_in
        self.outgoing = outgoing
        self.noise_scale = noise_scale

        self.generator = BigGAN.from_pretrained('biggan-deep-512')
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

            image.squeeze_(0)
            image = image.permute(1, 2, 0)

        self.outgoing.write(image)

    def cleanup(self):
        self.outgoing.cleanup_output()
        self.stft_in.cleanup_input()
        self.noise_in.cleanup_input()


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
        rgb = self.rgb_intensity.value

        cimage = cp.asarray(image)
        for idx in range(3):
            shift = np.random.randn(2).astype('float32') * rgb
            cimage[:, :, idx] = cpi.shift(cimage[:, :, idx], shift)

        # cimage = (cimage + 1) / 2
        # cimage = cimage * 255
        # nimage = cp.asnumpy(cimage).astype('uint8')

        self.outgoing.write(cimage)

    def cleanup(self):
        self.outgoing.cleanup_output()
        self.incoming.cleanup_input()


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

    def cleanup(self):
        self.incoming.cleanup_input()


@contextmanager
def cuda_activate(img):
    """Context manager simplifying use of pycuda.gl.RegisteredImage"""
    mapping = img.map()
    yield mapping.array(0, 0)
    mapping.unmap()


def create_shared_texture(map_flags=graphics_map_flags.WRITE_DISCARD, dtype=np.uint8):
    tex = np.zeros((1920, 1080, 4), dtype).view(gloo.Texture2D)
    tex.activate()  # force gloo to create on GPU
    tex.deactivate()
    cuda_buffer = pycuda.gl.RegisteredImage(
        int(tex.handle), tex.target, map_flags)
    return tex, cuda_buffer


class InteropDisplay(BaseNode):
    vertex = """
    uniform float scale;
    attribute vec2 position;
    attribute vec2 texcoord;
    varying vec2 v_texcoord;
    void main()
    {
        v_texcoord = texcoord;
        gl_Position = vec4(scale*position, 0.0, 1.0);
    } """
    fragment = """
    uniform sampler2D tex;
    varying vec2 v_texcoord;
    void main()
    {
        gl_FragColor = texture2D(tex, v_texcoord);
    } """

    def __init__(self, incoming: Edge, exit_event: mp.Event):
        super().__init__()
        self.incoming = incoming
        self.exit_event = exit_event

    def setup(self):
        self.window = app.Window(width=1920, height=1080, fullscreen=False, decoration=True)
        self.window.event(self.on_draw)

        import pycuda.gl.autoinit
        import pycuda.gl

        self.state = cp.ones((1920, 1080, 4))
        tex, self.cuda_buffer = create_shared_texture()

        self.screen = gloo.Program(InteropDisplay.vertex, InteropDisplay.fragment, count=4)
        self.screen['position'] = [(-1, -1), (-1, +1), (+1, -1), (+1, +1)]
        self.screen['texcoord'] = [(0, 0), (0, 1), (1, 0), (1, 1)]
        self.screen['scale'] = 1.0
        self.screen['tex'] = tex

        self.backend = app.__backend__
        self.clock = app.__init__(backend=self.backend)

    def on_draw(self, dt):
        self.window.set_title(f'{self.window.fps:.2f}')

        tex = self.screen['tex']
        self.state[:, :, :3] = self.incoming.read()
        if self.state is None:
            return

        tensor = (self.state + 1.) / 2 * 255
        tensor = cp.clip(tensor, 0, 255).astype(cp.uint8)
        tensor: cp.ndarray = cp.ascontiguousarray(tensor)

        assert tex.nbytes == tensor.nbytes
        with cuda_activate(self.cuda_buffer) as ary:
            cpy = pycuda.driver.Memcpy2D()
            cpy.set_src_device(tensor.data.ptr)
            cpy.set_dst_array(ary)
            cpy.width_in_bytes = cpy.src_pitch = cpy.dst_pitch = tex.nbytes // 1080
            cpy.height = 1080
            cpy(aligned=False)
            torch.cuda.synchronize()

        self.window.clear()
        self.screen.draw(gl.GL_TRIANGLE_STRIP)

    def run(self):
        if not self.backend.process(self.clock.tick()):
            self.exit_event.set()

    def cleanup(self):
        self.incoming.cleanup_input()
