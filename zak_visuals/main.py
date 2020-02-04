import ctypes
import signal
import logging

from torch import multiprocessing as mp

from zak_visuals.nodes import AudioProcessor, BIGGAN, ImageFX, InteropDisplay, NoiseGenerator, LabelGenerator
from zak_visuals.nodes import JACKInput, OSCServer
from zak_visuals.nodes.base_nodes import Edge

logger = mp.log_to_stderr()
logger.setLevel(logging.INFO)


class App:
    def __init__(self):
        mp.set_start_method('spawn', force=True)
        self.exit = mp.Event()

        self.manager = mp.Manager()
        params = self.manager.dict()
        params['rgb'] = 0.
        params['stft_scale'] = 0.
        params['animate_noise'] = False
        self.osc_server = OSCServer(self.exit, params=params)

        self.buffer = mp.Array(ctypes.c_float, 2048)
        self.cqt = Edge()
        self.noise = Edge()
        self.label = Edge()
        self.image = Edge()
        self.imfx = Edge()

        self.jack_input = JACKInput(outgoing=self.buffer)
        self.audio_processor = AudioProcessor(incoming=self.buffer, outgoing=self.cqt)
        # self.image_generator = PGGAN(incoming=self.cqt, outgoing=self.image)
        self.noise_generator = NoiseGenerator(outgoing=self.noise, params=params)
        self.label_generator = LabelGenerator(outgoing=self.label)
        self.image_generator = BIGGAN(stft_in=self.cqt, noise_in=self.noise, label_in=self.label,
                                      outgoing=self.image, params=params)
        self.image_fx = ImageFX(incoming=self.image, outgoing=self.imfx, params=params)
        self.image_display = InteropDisplay(incoming=self.imfx, exit_event=self.exit)

    def run(self):
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        self.osc_server.start()
        self.jack_input.start()
        self.audio_processor.start()
        self.noise_generator.start()
        self.label_generator.start()
        self.image_generator.start()
        self.image_fx.start()
        self.image_display.start()
        signal.signal(signal.SIGINT, self.on_keyboard_interrupt)
        self.exit.wait()
        self.exit_handler()

    def exit_handler(self):
        self.jack_input.stop()
        self.audio_processor.stop()
        self.noise_generator.stop()
        self.label_generator.stop()
        self.image_generator.stop()
        self.image_fx.stop()
        self.image_display.stop()
        self.osc_server.stop()

    def on_keyboard_interrupt(self, sig, _):
        print("KeyboardInterrupt (ID: {}) has been caught. Cleaning up...".format(sig))
        self.exit_handler()
        exit(0)


def main():
    app = App()
    app.run()


if __name__ == '__main__':
    main()
