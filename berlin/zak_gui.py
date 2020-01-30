import multiprocessing as mp
import queue
import signal
import threading

import cv2
import jack
import librosa
import numpy as np
import torch
from pythonosc import dispatcher
from pythonosc import osc_server
from scipy.interpolate import interp1d

from berlin.pg_gan.model import Generator

CHECKPOINT_PATH = 'saves/zak1.1/Models/Gs_nch-4_epoch-347.pth'


def resample(x, n, kind='cubic'):
    f = interp1d(np.linspace(0, 1, x.size), x, kind)
    return f(np.linspace(0, 1, n))


# TODO: Maybe ImageDisplay should be in the main thread
# TODO: Processes stil don't terminate in a sane way. Improved a bit I guess.
# TODO: OSCParameters should be a global shared state
class BaseProcess:
    def __init__(self, incoming=None, outgoing=None, osc_params=None):
        if not incoming and not outgoing:
            raise ValueError('All processes must have at least an input or an output!')

        self.incoming = incoming
        self.outgoing = outgoing
        self.osc_params = osc_params

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
        signal.signal(signal.SIGINT, signal.SIG_IGN)

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


class App:
    def __init__(self):
        self.manager = mp.Manager()
        self.osc_params = self.manager.Namespace()
        self.osc_params.rgb_intensity = 0.0
        self.osc_server = OSCServer(self.osc_params)

        self.buffer = mp.Queue(maxsize=1)
        self.cqt = mp.Queue(maxsize=1)
        self.image = mp.Queue(maxsize=1)
        self.imfx = mp.Queue(maxsize=1)

        self.jack_input = JACKInput(self.buffer)
        self.audio_processor = AudioProcessor(incoming=self.buffer, outgoing=self.cqt)
        self.image_generator = ImageGenerator(incoming=self.cqt, outgoing=self.image)
        self.image_fx = ImageFX(incoming=self.image, outgoing=self.imfx, osc_params=self.osc_params)
        self.image_display = ImageDisplay(incoming=self.imfx)

        self.exit = threading.Event()

    def run(self):
        self.osc_server.start()
        self.jack_input.start()
        self.audio_processor.start()
        self.image_generator.start()
        self.image_fx.start()
        self.image_display.start()
        self.exit.wait()

    def exit_handler(self, signals, frame_type):
        self.jack_input.stop()
        self.audio_processor.stop()
        self.image_generator.stop()
        self.image_fx.stop()
        self.image_display.stop()
        self.osc_server.stop()
        self.exit.set()


class JACKInput:
    def __init__(self, buffer: mp.Queue):
        self.buffer = buffer
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


class AudioProcessor(BaseProcess):
    def run(self):
        try:
            buffer = self.incoming.get(timeout=1)
        except queue.Empty:
            return
        stft = librosa.stft(buffer, n_fft=2048, hop_length=2048, center=False, window='boxcar')
        stft = np.abs(stft).squeeze(1).astype('float32')
        stft = 2 * stft / 2048
        # stft = resample(stft, 128)
        stft = stft[0:128]
        self.outgoing.put(stft)


class ImageGenerator(BaseProcess):
    def setup(self):
        if torch.cuda.is_available():
            self.generator: Generator = torch.load(CHECKPOINT_PATH)
        else:
            self.generator: Generator = torch.load(CHECKPOINT_PATH, map_location=torch.device('cpu'))

    def run(self):
        try:
            stft = self.incoming.get(timeout=1)
        except queue.Empty:
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

        self.outgoing.put(image)


class ImageFX(BaseProcess):
    def run(self):
        try:
            image = self.incoming.get(timeout=1)
        except queue.Empty:
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

        self.outgoing.put(image)


# processes were stuck on empty queues
# maybe ques should modify some global values instead of being directly accessed by processes
class ImageDisplay(BaseProcess):
    def setup(self):
        cv2.namedWindow('frame', cv2.WINDOW_FREERATIO)
        cv2.setWindowProperty('frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    def run(self):
        try:
            image = self.incoming.get(timeout=1)
        except queue.Empty:
            return

        cv2.imshow('frame', image)
        cv2.waitKey(1)


class OSCServer:
    def __init__(self, osc_params):
        super().__init__()
        self.processor = threading.Thread(target=self.process)
        self.osc_parameters = osc_params
        self.dispatcher = dispatcher.Dispatcher()
        self.server = osc_server.ThreadingOSCUDPServer(('0.0.0.0', 8000), self.dispatcher)

        self.dispatcher.map('/visuals/patch', self.rgb_intensity)

    def rgb_intensity(self, addr, value):
        self.osc_parameters.rgb_intensity = value * 50

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
    signal.signal(signal.SIGINT, app.exit_handler)
    app.run()


if __name__ == '__main__':
    main()
