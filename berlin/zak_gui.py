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

from berlin.pg_gan.model import Generator


# TODO: OSCClient is in fact an OSCServer
# TODO: OSCClient shutdown deadlocks so we terminate. Fix it.
# TODO: Jack input should be its own thread
# TODO: Maybe ImageDisplay should be in the main thread
class BaseProcess:
    def __init__(self):
        self.processor = mp.Process(target=self.process)
        self.exit = mp.Event()

    def start(self):
        self.processor.start()

    def stop(self):
        print(f'Exiting {self.__class__.__name__}.')
        self.exit.set()
        self.processor.join()

    def process(self):
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        self.setup()
        while not self.exit.is_set():
            self.run()

        print(f'{self.__class__.__name__} is kill!')

    def setup(self):
        pass

    def run(self):
        raise NotImplemented


class App:
    def __init__(self):
        self.manager = mp.Manager()
        self.osc_parameters = self.manager.Namespace()
        self.osc_parameters.rgb_intensity = 0.0
        self.osc_client = OSCClient(self.osc_parameters)

        self.buffer = mp.Queue(maxsize=1)
        self.cqt = mp.Queue(maxsize=1)
        self.image = mp.Queue(maxsize=1)
        self.imfx = mp.Queue(maxsize=1)

        self.jack_input = JACKInput(self.buffer)
        self.audio_processor = AudioProcessor(self.buffer, self.cqt)
        self.image_generator = ImageGenerator(self.cqt, self.image)
        self.image_fx = ImageFX(self.image, self.imfx, self.osc_parameters)
        self.image_display = ImageDisplay(self.imfx)

    def run(self):
        self.osc_client.start()

        self.audio_processor.start()
        self.image_generator.start()
        self.image_fx.start()
        self.image_display.start()

        # jack_input is the last process to start
        self.jack_input.start()

    def exit_handler(self, signals, frame_type):
        self.jack_input.stop()
        self.audio_processor.stop()
        self.image_generator.stop()
        self.image_fx.stop()
        self.image_display.stop()
        self.osc_client.stop()


class JACKInput:
    def __init__(self, buffer: mp.Queue):
        self.buffer = buffer
        self.client = jack.Client('Zak')
        self.inport: jack.OwnPort = self.client.inports.register('input_1')
        self.exit = threading.Event()
        self.client.set_process_callback(self.read_buffer)

    def read_buffer(self, frames: int):
        assert frames == self.client.blocksize
        buffer = self.inport.get_array().astype('float32')
        self.buffer.put(buffer)

    def start(self):
        sysport: jack.Port = self.client.get_ports(is_audio=True, is_output=True, is_physical=True)[0]

        with self.client:
            self.client.connect(sysport, self.inport)

            self.exit.wait()
            print(f'Exiting {self.__class__.__name__} process...')

    def stop(self):
        self.exit.set()


class AudioProcessor(BaseProcess):
    def __init__(self, buffer: mp.Queue, cqt: mp.Queue):
        super().__init__()
        self.buffer = buffer
        self.cqt = cqt

    def run(self):
        try:
            buffer = self.buffer.get(block=False)
        except queue.Empty:
            return
        cqt = librosa.stft(buffer, n_fft=2048, hop_length=2048, center=False)
        self.cqt.put(cqt)


class ImageGenerator(BaseProcess):
    def __init__(self, cqt: mp.Queue, image: mp.Queue):
        super().__init__()
        self.cqt = cqt
        self.image = image
        self.generator = None

    def setup(self):
        self.generator: Generator = torch.load('saves/zak1.1/Models/Gs_nch-4_epoch-347.pth').cuda()

    def run(self):
        try:
            cqt = self.cqt.get(block=False)
        except queue.Empty:
            return
        features = np.zeros((1, 128, 1, 1), dtype='float32')
        features[0, :, 0, 0] = np.abs(cqt[:128, 0]).astype('float32')
        features = torch.from_numpy(features).cuda()
        with torch.no_grad():
            image = self.generator(features)
            image = torch.nn.functional.interpolate(image, (1920, 1080))

            image = (image + 1) / 2
            image = image * 255
            image.squeeze_(0)
            image = image.permute(1, 2, 0)
            image = image.expand(1920, 1080, 3)

            image = image.cpu().numpy().astype('uint8')

        self.image.put(image)


class ImageFX(BaseProcess):
    def __init__(self, image: mp.Queue, imfx: mp.Queue, osc_parameters):
        super().__init__()
        self.image = image
        self.imfx = imfx
        self.osc_parameters = osc_parameters

    def run(self):
        try:
            image = self.image.get(block=False)
        except queue.Empty:
            return

        base_points = np.float32([[420, 1080], [420, 0], [1500, 0]])
        try:
            rgb = self.osc_parameters.rgb_intensity
        except BrokenPipeError:
            return
        for idx in range(3):
            shift = np.random.normal(0, rgb, (3, 2)).astype('float32')
            m = cv2.getAffineTransform(base_points, base_points + shift)
            image[:, :, idx] = cv2.warpAffine(image[:, :, idx], m, (1080, 1920))

        self.imfx.put(image)


# processes were stuck on empty queues
# maybe ques should modify some global values instead of being directly accessed by processes
class ImageDisplay(BaseProcess):
    def __init__(self, imfx: mp.Queue):
        super().__init__()
        self.imfx = imfx

    def setup(self):
        cv2.namedWindow('frame', cv2.WINDOW_FREERATIO)
        cv2.setWindowProperty('frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    def run(self):
        try:
            imfx = self.imfx.get(block=False)
        except queue.Empty:
            return

        cv2.imshow('frame', imfx)
        cv2.waitKey(1)


class OSCClient:
    def __init__(self, osc_parameters):
        super().__init__()
        self.processor = mp.Process(target=self.process)
        self.osc_parameters = osc_parameters
        self.dispatcher = dispatcher.Dispatcher()
        self.server = osc_server.ThreadingOSCUDPServer(('0.0.0.0', 8000), self.dispatcher)

        self.dispatcher.map('/visuals/patch', self.rgb_intensity)

    def rgb_intensity(self, addr, value):
        self.osc_parameters.rgb_intensity = value * 50

    def process(self):
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        self.server.serve_forever()
        print(f'{self.__class__.__name__} is kill!')

    def start(self):
        self.processor.start()

    def stop(self):
        print(f'Exiting {self.__class__.__name__}')
        self.processor.terminate()


def main():
    app = App()
    signal.signal(signal.SIGINT, app.exit_handler)
    app.run()


if __name__ == '__main__':
    main()
