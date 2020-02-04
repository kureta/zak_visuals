import threading

import jack
from pythonosc import dispatcher, osc_server
from torch import multiprocessing as mp


class JACKInput(threading.Thread):
    def __init__(self, outgoing: mp.Array):
        super().__init__()
        self.buffer = outgoing
        self.client = jack.Client('Zak')
        self.inport: jack.OwnPort = self.client.inports.register('input_1')
        self.exit = threading.Event()
        self.client.set_process_callback(self.read_buffer)

    def read_buffer(self, frames: int):
        assert frames == self.client.blocksize
        self.buffer[:] = self.inport.get_array()[:]

    def run(self):
        sysport: jack.Port = self.client.get_ports(is_audio=True, is_output=True, is_physical=True)[0]

        with self.client:
            self.client.connect(sysport, self.inport)
            self.exit.wait()

    def join(self, **kwargs):
        self.exit.set()
        super(JACKInput, self).join(**kwargs)


class OSCServer(threading.Thread):
    def __init__(self, exit_app: mp.Event, params: dict):
        super().__init__()
        self.exit_app = exit_app
        self.dispatcher = dispatcher.Dispatcher()
        self.server = osc_server.ThreadingOSCUDPServer(('0.0.0.0', 8000), self.dispatcher)

        self.params = params

        self.dispatcher.map('/2/rgb', self.on_rgb_intensity)
        self.dispatcher.map('/2/noise', self.on_noise_scale)
        self.dispatcher.map('/2/animate_noise', self.on_animate_noise)
        self.dispatcher.map('/2/label_group/*', self.on_label_group)
        self.dispatcher.map('/2/randomize_label', self.on_randomize_label)
        self.dispatcher.map('/2/quit', self.quit)
        self.dispatcher.set_default_handler(self.on_unknown_message)

    def quit(self, addr, value):
        self.exit_app.set()

    @staticmethod
    def on_unknown_message(addr, *values):
        print(f'addr: {addr}', f'values: {values}')

    def on_rgb_intensity(self, addr, value):
        self.params['rgb'].value = value

    def on_noise_scale(self, addr, value):
        self.params['stft_scale'].value = value

    def on_animate_noise(self, addr, value):
        self.params['animate_noise'].value = value

    def on_label_group(self, addr, value):
        if value > 0.:
            idx = addr.split('/')[3]
            self.params['label_group'].value = int(idx) - 1

    def on_randomize_label(self, addr, value):
        self.params['randomize_label'].value = value

    def run(self):
        self.server.serve_forever()

    def join(self, **kwargs):
        self.server.shutdown()
        super(OSCServer, self).join(**kwargs)
