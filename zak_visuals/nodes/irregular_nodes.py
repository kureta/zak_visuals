import threading
from multiprocessing import managers

import jack
from pythonosc import dispatcher, osc_server
from torch import multiprocessing as mp


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