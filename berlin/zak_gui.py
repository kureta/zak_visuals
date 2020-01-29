import multiprocessing as mp
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


class BaseProcess:
    def __init__(self):
        self.processor = mp.Process(target=self.process)
        self.exit = mp.Event()

    def start(self):
        self.processor.start()

    def stop(self):
        self.exit.set()
        self.processor.terminate()

    def process(self):
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        self.setup()
        while not self.exit.is_set():
            self.run()

        print(f'Exiting {self.__class__.__name__} process.')

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

    def exit(self, signals, frame_type):
        self.osc_client.stop()

        self.image_display.stop()
        self.image_fx.stop()
        self.image_generator.stop()
        self.audio_processor.stop()
        self.jack_input.stop()


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
        buffer = self.buffer.get()
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
        cqt = self.cqt.get()
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
        image = self.image.get()

        base_points = np.float32([[420, 1080], [420, 0], [1500, 0]])
        rgb = self.osc_parameters.rgb_intensity
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
        imfx = self.imfx.get()

        cv2.imshow('frame', imfx)
        cv2.waitKey(1)


class OSCClient(BaseProcess):
    def __init__(self, osc_parameters):
        super().__init__()
        self.osc_parameters = osc_parameters
        self.dispatcher = dispatcher.Dispatcher()
        self.server = osc_server.ThreadingOSCUDPServer(('0.0.0.0', 8000), self.dispatcher)

        self.dispatcher.map('/visuals/patch', self.rgb_intensity)

    def rgb_intensity(self, addr, value):
        self.osc_parameters.rgb_intensity = value * 50

    def process(self):
        self.server.serve_forever()

    def stop(self):
        self.server.server_close()


def main():
    app = App()
    signal.signal(signal.SIGINT, app.exit)
    app.run()


if __name__ == '__main__':
    main()
