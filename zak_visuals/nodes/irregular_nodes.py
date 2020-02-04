import threading

import jack
from pythonosc import dispatcher, osc_server
from torch import multiprocessing as mp


class JACKInput:
    def __init__(self, outgoing: mp.Array):
        self.buffer = outgoing
        self.client = jack.Client('Zak')
        self.inport: jack.OwnPort = self.client.inports.register('input_1')
        self.exit = threading.Event()
        self.client.set_process_callback(self.read_buffer)
        self.processor = threading.Thread(target=self.process)

    def read_buffer(self, frames: int):
        assert frames == self.client.blocksize
        self.buffer[:] = self.inport.get_array()[:]

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


class OSCServer:
    def __init__(self, exit_event: mp.Event, params: dict):
        super().__init__()
        self.processor = threading.Thread(target=self.process)
        self.exit_event = exit_event
        self.dispatcher = dispatcher.Dispatcher()
        self.server = osc_server.ThreadingOSCUDPServer(('0.0.0.0', 8000), self.dispatcher)

        self.params = params

        self.dispatcher.map('/2/rgb', self.on_rgb_intensity)
        self.dispatcher.map('/2/noise', self.on_noise_scale)
        self.dispatcher.map('/2/animate_noise', self.on_animate_noise)
        self.dispatcher.map('/2/quit', self.quit)
        self.dispatcher.set_default_handler(self.on_unknown_message)

    def quit(self, addr, value):
        self.exit_event.set()

    @staticmethod
    def on_unknown_message(addr, *values):
        print(f'addr: {addr}', f'values: {values}')

    def on_rgb_intensity(self, addr, value):
        self.params['rgb'] = value

    def on_noise_scale(self, addr, value):
        self.params['stft_scale'] = value

    def on_animate_noise(self, addr, value):
        self.params['animate_noise'] = value

    def process(self):
        self.server.serve_forever()

    def start(self):
        self.processor.start()

    def stop(self):
        print(f'Exiting {self.__class__.__name__}')
        self.server.shutdown()
        self.processor.join()
        print(f'{self.__class__.__name__} is kill!')
