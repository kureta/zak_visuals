import multiprocessing as mp
import queue
import threading
from multiprocessing import managers

import cv2
import jack
import librosa
import numpy as np
import torch
from pythonosc import dispatcher
from pythonosc import osc_server
from pytorch_pretrained_biggan import BigGAN, one_hot_from_names, truncated_noise_sample
from scipy.interpolate import interp1d

from berlin.pg_gan.model import Generator

CHECKPOINT_PATH = 'saves/zak1.1/Models/Gs_nch-4_epoch-347.pth'


def resample(x, n, kind='cubic', axis=0):
    f = interp1d(np.linspace(0, 1, x.shape[axis]), x, kind, axis=axis)
    return f(np.linspace(0, 1, n))


# TODO: Rename all the things!
# TODO: We will need multi-input processor nodes.
# TODO: Animate latent space exploration in BigGAN
# TODO: Create 2 OSCClients
#       One will relay unhandled messages to Onur
#       Other will send initialization values to TouchOSC
class BaseNode:
    def __init__(self):
        self.processor = mp.Process(target=self.process)
        self.exit = mp.Event()

    def start(self):
        self.processor.start()

    def stop(self):
        print(f'Exiting {self.__class__.__name__}.')
        self.exit.set()
        self.processor.join()
        print(f'{self.__class__.__name__} is kill!')

    def process(self):
        self.setup()
        while not self.exit.is_set():
            self.run()
        self.teardown()

    def setup(self):
        pass

    def run(self):
        raise NotImplemented

    def teardown(self):
        pass


class InputNode(BaseNode):
    def __init__(self, outgoing: mp.Queue):
        super().__init__()
        self._outgoing = outgoing

    @property
    def outgoing(self):
        raise PermissionError('Not allowed to read from output!')

    @outgoing.setter
    def outgoing(self, value):
        self._outgoing.put(value)

    def run(self):
        raise NotImplemented


class OutputNode(BaseNode):
    def __init__(self, incoming: mp.Queue):
        super().__init__()
        self._incoming = incoming

    @property
    def incoming(self):
        try:
            value = self._incoming.get(timeout=1)
        except queue.Empty:
            value = None
        return value

    @incoming.setter
    def incoming(self, value):
        raise PermissionError('Not allowed to write to input!')

    def run(self):
        raise NotImplemented


class ProcessorNode(BaseNode):
    def __init__(self, incoming: mp.Queue, outgoing: mp.Queue):
        super().__init__()
        self._incoming = incoming
        self._outgoing = outgoing

    @property
    def incoming(self):
        try:
            value = self._incoming.get(timeout=1)
        except queue.Empty:
            value = None
        return value

    @incoming.setter
    def incoming(self, value):
        raise PermissionError('Not allowed to write to input!')

    @property
    def outgoing(self):
        raise PermissionError('Not allowed to read from output!')

    @outgoing.setter
    def outgoing(self, value):
        self._outgoing.put(value)

    def run(self):
        raise NotImplemented


class App:
    def __init__(self):
        self.exit = mp.Event()

        self.manager = mp.Manager()
        self.osc_params = self.manager.Namespace()
        self.osc_params.rgb_intensity = 0.0
        self.osc_params.scale = 1.0
        self.osc_server = OSCServer(self.osc_params, self.exit)

        self.buffer = mp.Queue(maxsize=1)
        self.cqt = mp.Queue(maxsize=1)
        self.image = mp.Queue(maxsize=1)
        self.imfx = mp.Queue(maxsize=1)

        self.jack_input = JACKInput(outgoing=self.buffer)
        self.audio_processor = AudioProcessor(incoming=self.buffer, outgoing=self.cqt)
        self.image_generator = AlternativeGenerator(incoming=self.cqt, outgoing=self.image, osc_params=self.osc_params)
        self.image_fx = ImageFX(incoming=self.image, outgoing=self.imfx, osc_params=self.osc_params)
        self.image_display = ImageDisplay(incoming=self.imfx, exit_event=self.exit)

    def run(self):
        self.osc_server.start()
        self.jack_input.start()
        self.audio_processor.start()
        self.image_generator.start()
        self.image_fx.start()
        self.image_display.start()
        self.exit.wait()
        self.exit_handler()

    def exit_handler(self):
        self.jack_input.stop()
        self.audio_processor.stop()
        self.image_generator.stop()
        self.image_fx.stop()
        self.image_display.stop()
        self.osc_server.stop()


class JACKInput:
    def __init__(self, outgoing: mp.Queue):
        self.buffer = outgoing
        self.client = jack.Client('Zak')
        self.inport: jack.OwnPort = self.client.inports.register('input_1')
        self.exit = threading.Event()
        self.client.set_process_callback(self.read_buffer)
        self.processor = threading.Thread(target=self.process)

    def read_buffer(self, frames: int):
        assert frames == self.client.blocksize
        buffer = self.inport.get_array().astype('float32')
        self.buffer.put(buffer)

    def process(self):
        sysport: jack.Port = self.client.get_ports(is_audio=True, is_output=True, is_physical=True)[0]

        with self.client:
            self.client.connect(sysport, self.inport)
            self.exit.wait()

    def start(self):
        self.processor.start()

    def stop(self):
        print(f'Exiting {self.__class__.__name__} process...')
        self.exit.set()
        self.processor.join()
        print(f'{self.__class__.__name__} is kill!')


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
        if torch.cuda.is_available():
            self.generator: Generator = torch.load(CHECKPOINT_PATH)
        else:
            self.generator: Generator = torch.load(CHECKPOINT_PATH, map_location=torch.device('cpu'))

    def run(self):
        stft = self.incoming
        if stft is None:
            return
        features = np.zeros((1, 128, 1, 1), dtype='float32')
        features[0, :, 0, 0] = stft
        features = torch.from_numpy(features)
        if torch.cuda.is_available():
            features = features.cuda()
        with torch.no_grad():
            image = self.generator(features)
            image = torch.nn.functional.interpolate(image, (1920, 1080))

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
        self.generator = BigGAN.from_pretrained('biggan-deep-512')

        labels = ['analog clock', 'disk brake', 'loudspeaker']
        class_vector = one_hot_from_names(labels)
        class_vector = resample(class_vector, (len(labels) - 1) * 64, kind='linear', axis=0).astype('float32')
        class_vector = np.expand_dims(class_vector, axis=1)

        noise_vector = truncated_noise_sample(truncation=0.4, batch_size=256)
        noise_vector = resample(noise_vector, 1024, axis=0).astype('float32')
        noise_vector = np.expand_dims(noise_vector, axis=1)
        print(noise_vector.shape)
        print(noise_vector.min(), noise_vector.max())
        self.idx = 0
        self.class_vector = torch.from_numpy(class_vector)
        self.noise_vector = torch.from_numpy(noise_vector)

        if torch.cuda.is_available():
            self.class_vector = self.class_vector.cuda()
            self.noise_vector = self.noise_vector.cuda()
            self.generator.cuda()

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
        if torch.cuda.is_available():
            features = features.cuda()
        features = features + self.noise_vector[self.idx]
        self.idx = (self.idx + 1) % 1024
        with torch.no_grad():
            image = self.generator(features, self.class_vector[self.idx % self.class_vector.shape[0]], 0.4)
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


# processes were stuck on empty queues
# maybe ques should modify some global values instead of being directly accessed by processes
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


class OSCServer:
    def __init__(self, osc_params: managers.Namespace, exit_event: mp.Event):
        super().__init__()
        self.processor = threading.Thread(target=self.process)
        self.osc_parameters = osc_params
        self.exit_event = exit_event
        self.dispatcher = dispatcher.Dispatcher()
        self.server = osc_server.ThreadingOSCUDPServer(('0.0.0.0', 8000), self.dispatcher)

        self.dispatcher.map('/visuals/patch', self.rgb_intensity)
        self.dispatcher.map('/visuals/smooth', self.scale)
        self.dispatcher.map('/visuals/run', self.quit)
        self.dispatcher.set_default_handler(self.unknown_message)

    def quit(self, addr, value):
        self.exit_event.set()

    @staticmethod
    def unknown_message(addr, values):
        print(f'addr: {addr}', f'values: {values}')

    def rgb_intensity(self, addr, value):
        self.osc_parameters.rgb_intensity = value * 50

    def scale(self, addr, value):
        self.osc_parameters.scale = value * 99 + 1

    def process(self):
        self.server.serve_forever()

    def start(self):
        self.processor.start()

    def stop(self):
        print(f'Exiting {self.__class__.__name__}')
        self.server.shutdown()
        self.processor.join()
        print(f'{self.__class__.__name__} is kill!')


def main():
    app = App()
    app.run()


if __name__ == '__main__':
    main()
