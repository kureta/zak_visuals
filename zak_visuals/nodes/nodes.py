from contextlib import contextmanager

import librosa
import moderngl as mgl
import moderngl_window as mglw
import numpy as np
import pycuda.driver
import torch
from moderngl_window import geometry
from pycuda.gl import graphics_map_flags
from pyglet.gl import GL_TEXTURE_2D
from torch import multiprocessing as mp
from torch.nn import functional as F

import stylegan2
from berlin.pg_gan.utils import hypersphere
from zak_visuals.nodes.base_nodes import BaseNode
from zak_visuals.utils.constants import label_groups

DEVICE = 'cuda:0'


class AudioProcessor(BaseNode):
    def __init__(self, incoming: mp.Array, outgoing: mp.Array, rms: mp.Value, pause_event: mp.Event):
        super().__init__(pause_event=pause_event)
        self.incoming = incoming
        self.outgoing = outgoing
        self.rms = rms
        self.count = 0
        self.mean = 0
        self.std = 0
        self.rms_count = 0
        self.rms_mean = 0
        self.rms_std = 0
        self.epsilon = 1e-9

    def setup(self):
        self.buffer = np.ndarray((2048,), dtype='float32', buffer=self.incoming)

    def task(self):
        stft = librosa.stft(self.buffer, n_fft=2048, hop_length=2048, center=False, window='boxcar')
        stft = np.abs(stft)
        rms = librosa.feature.rms(S=stft, frame_length=2048, hop_length=2048, center=False)
        rms = rms.squeeze(1).astype('float32')
        stft = stft.squeeze(1).astype('float32')
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

        new_count = self.rms_count + 1
        new_mean = (self.rms_mean * self.rms_count + rms) / new_count
        diff = rms - new_mean
        new_std = np.sqrt((np.square(self.rms_std) * self.rms_count + np.dot(diff, diff)) / new_count)

        self.rms_mean = new_mean
        self.rms_std = new_std
        self.rms_count = new_count

        self.outgoing[:] = (stft[:128] - self.mean) / max(self.epsilon, self.std)
        self.rms.value = (rms - self.rms_mean) / max(self.epsilon, self.rms_std)


# TODO: User real time in all animated values
class NoiseGenerator(BaseNode):
    def __init__(self, outgoing: mp.Queue, params: dict, pause_event: mp.Event):
        super().__init__(pause_event=pause_event)
        self.outgoing = outgoing
        self.params = params
        self.num_frames = 128
        self.speed = 1
        self.sampling_radius = 1.
        self.frame = 0
        self.idx = 0
        self.moving = False

    def swap(self):
        self.buffers[:] = self.buffers[::-1]

    def set_start(self, motion, previous_motion):
        motion[0, :, :] = previous_motion[-1, :, :]

    def create_motion(self, motion):
        motion[-1, :, :] = hypersphere(torch.randn(1, 128, device=DEVICE), radius=self.sampling_radius)
        motion[:] = F.interpolate(torch.stack((motion[0], motion[-1])).permute(1, 2, 0),
                                  self.num_frames, mode='linear', align_corners=True).permute(2, 0, 1)

    def setup(self):
        torch.autograd.set_grad_enabled(False)
        self.motion_1 = torch.zeros(self.num_frames, 1, 128, device=DEVICE)
        self.motion_2 = torch.zeros(self.num_frames, 1, 128, device=DEVICE)
        self.buffers = [self.motion_1, self.motion_2]
        self.set_start(self.buffers[0], self.buffers[1])

    def restart(self):
        # self.sampling_radius = self.params['noise_std'].value * 12. + 0.01
        self.moving = True
        self.create_motion(self.buffers[0])
        self.set_start(self.buffers[1], self.buffers[0])

    def task(self):
        self.speed = int(self.params['noise_speed'].value * 10)

        if self.frame >= self.num_frames - 1:
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
        self.previous_category = 0

    def swap(self):
        self.buffers[:] = self.buffers[::-1]

    def get_label(self):
        category = self.params['label_group'].value
        if category != self.previous_category:
            self.idx = 0
            self.previous_category = category
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


# TODO: Buffering and batch generation speeds up the process but introduces visible latency
class StyleGAN2(BaseNode):
    def __init__(self, pause_event: mp.Event, noise: mp.Queue, stft_in: mp.Array, outgoing: mp.Queue, params: dict):
        super().__init__(pause_event=pause_event)
        self.noise = noise
        self.stft_in = stft_in
        self.outgoing = outgoing
        self.params = params

    # TODO: interpolate layer weights between different pretrained models
    #       it is harder than I thought. Later.
    # TODO: There are 14 noise input layers. Lower to higher frequency features.
    #       Can send STFT bands to corresponding layers. Control noise animations for each layer separately.
    # TODO: get rid of other generators. Rethink architecture, keeping nodez UI in mind.
    def setup(self):
        checkpoint_path = '/home/kureta/Documents/other-repos/stylegan2_pytorch/pretrained/cats/Gs.pth'
        self.generator = stylegan2.models.load(checkpoint_path)
        self.generator.static_noise()
        self.generator.eval()
        self.generator.to(DEVICE)

        torch.autograd.set_grad_enabled(False)
        self.stft = np.ndarray((128,), dtype='float32', buffer=self.stft_in)

        self.first = torch.randn((1, 2, 512), device=DEVICE)
        self.last = torch.randn((1, 11, 512), device=DEVICE)

    def task(self):
        noise = self.noise.get()
        noise = noise.repeat(1, 4)
        noise.unsqueeze_(0)
        noise = torch.cat([self.first, noise, self.last], dim=1)
        scale = self.params['stft_scale'].value * 20.

        features = torch.from_numpy(self.stft).to(DEVICE).unsqueeze(0).repeat(noise.shape[0], 4) * scale
        noise[:, 3, :] = features
        image = self.generator(noise)
        image.squeeze_(0)
        image = image.permute(1, 2, 0)

        self.outgoing.put(image)


@contextmanager
def cuda_activate(img):
    """Context manager simplifying use of pycuda.gl.RegisteredImage"""
    mapping = img.map()
    yield mapping.array(0, 0)
    mapping.unmap()


# TODO: Clean-up code
# TODO: Add back preview window using shared context
# TODO: Add shader effects
class InteropDisplay(BaseNode):
    def __init__(self, incoming: mp.Queue, exit_app: mp.Event):
        super().__init__()
        self.incoming = incoming
        self.exit_app = exit_app

    def create_shared_texture(self, w, h, c=4,
                              map_flags=graphics_map_flags.WRITE_DISCARD):
        """Create and return a Texture2D with gloo and pycuda views."""
        tex = self.window.ctx.texture((w, h), c)
        tex.filter = mgl.LINEAR, mgl.LINEAR
        tex.use(location=0)
        cuda_buffer = pycuda.gl.RegisteredImage(
            tex.glo, GL_TEXTURE_2D, map_flags)
        return tex, cuda_buffer

    def setup(self):
        win_str = 'moderngl_window.context.pyglet.Window'
        win_cls = mglw.get_window_cls(win_str)
        self.window = win_cls(
            title='Zak Visuals',
            gl_version=(4, 6),
            size=(1920, 1080),
            resizable=False,
            fullscreen=True,
            vsync=True,
            aspect_ratio=16 / 9,
            samples=4
        )
        mglw.activate_context(ctx=self.window.ctx)

        with open('shaders/screen/vertex.glsl') as f:
            vertex_shader = f.read()
        with open('shaders/screen/fragment.glsl') as f:
            fragment_shader = f.read()

        self.prog = self.window.ctx.program(
            vertex_shader=vertex_shader,
            fragment_shader=fragment_shader
        )

        import pycuda.gl.autoinit  # noqa
        self.state = torch.ones((256, 256, 4), dtype=torch.float32, device='cuda', requires_grad=False)
        texture, self.cuda_buffer = self.create_shared_texture(256, 256, 4)
        self.quad_fs = geometry.quad_fs()

    def task(self):
        tensor: torch.Tensor = self.incoming.get()
        tensor = (tensor + 1) / 2
        self.state[..., :3] = tensor
        tensor = self.state * 255.
        tensor = torch.clip(tensor, 0, 255)
        tensor = tensor.byte().contiguous().data  # convert to ByteTensor

        w = 256
        h = 256
        with cuda_activate(self.cuda_buffer) as ary:
            cpy = pycuda.driver.Memcpy2D()
            cpy.set_src_device(tensor.data_ptr())
            cpy.set_dst_array(ary)
            cpy.width_in_bytes = cpy.src_pitch = cpy.dst_pitch = w * 4
            cpy.height = h
            cpy(aligned=False)
            torch.cuda.synchronize()

        self.window.clear()
        self.quad_fs.render(self.prog)
        self.window.swap_buffers()
        if self.window.is_closing:
            self.exit_app.set()
