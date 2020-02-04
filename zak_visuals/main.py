import ctypes
import logging
import signal

from multiprocessing.sharedctypes import RawValue, RawArray
from torch import multiprocessing as mp

from zak_visuals.nodes import AudioProcessor, BIGGAN, ImageFX, InteropDisplay, NoiseGenerator, LabelGenerator, PGGAN
from zak_visuals.nodes import JACKInput, OSCServer

logger = mp.log_to_stderr()
logger.setLevel(logging.ERROR)


class App:
    def __init__(self):
        mp.set_start_method('spawn', force=True)
        self.exit = mp.Event()

        self.pause_pggan = mp.Event()
        self.pause_biggan = mp.Event()

        rgb = RawValue(ctypes.c_float)
        stft_scale = RawValue(ctypes.c_float)
        animate_noise = RawValue(ctypes.c_float)
        randomize_label = RawValue(ctypes.c_float)
        label_group = RawValue(ctypes.c_uint8)
        label_speed = RawValue(ctypes.c_float)
        noise_speed = RawValue(ctypes.c_float)
        noise_std = RawValue(ctypes.c_float)
        params = {
            'rgb': rgb,
            'stft_scale': stft_scale,
            'animate_noise': animate_noise,
            'randomize_label': randomize_label,
            'label_group': label_group,
            'label_speed': label_speed,
            'noise_speed': noise_speed,
            'noise_std': noise_std,
            'pause_gans': [self.pause_pggan, self.pause_biggan],
        }
        rgb.value, stft_scale.value, animate_noise.value, randomize_label.value = 0., 0., 0., 0.
        label_speed.value, noise_speed.value, noise_std.value = 0., 0., 0.
        label_group.value = 0
        self.osc_server = OSCServer(self.exit, params=params)

        self.buffer = RawArray(ctypes.c_float, 2048)
        self.stft = RawArray(ctypes.c_float, 128)
        self.noise = mp.Queue(maxsize=1)
        self.label = mp.Queue(maxsize=1)
        self.image = mp.Queue(maxsize=1)
        self.imfx = mp.Queue(maxsize=1)

        self.jack_input = JACKInput(outgoing=self.buffer)
        self.audio_processor = AudioProcessor(incoming=self.buffer, outgoing=self.stft)
        self.image_generator_2 = PGGAN(pause_event=self.pause_pggan, incoming=self.stft, noise=self.noise,
                                       outgoing=self.image, params=params)
        self.noise_generator = NoiseGenerator(outgoing=self.noise, params=params)
        self.label_generator = LabelGenerator(outgoing=self.label, params=params)
        self.image_generator = BIGGAN(stft_in=self.stft, noise_in=self.noise, label_in=self.label,
                                      outgoing=self.image, params=params, pause_event=self.pause_biggan)
        self.image_fx = ImageFX(incoming=self.image, outgoing=self.imfx, params=params)
        self.image_display = InteropDisplay(incoming=self.imfx, exit_app=self.exit)

    def run(self):
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        self.osc_server.start()
        self.jack_input.start()

        self.audio_processor.start()
        self.noise_generator.start()
        self.label_generator.start()
        self.image_generator_2.start()
        self.image_generator.start()
        self.image_fx.start()
        self.image_display.start()

        signal.signal(signal.SIGINT, self.on_keyboard_interrupt)

        self.exit.wait()
        self.exit_handler()

    def exit_handler(self):
        self.audio_processor.kill()
        self.noise_generator.kill()
        self.label_generator.kill()
        self.image_generator.kill()
        self.image_generator_2.kill()
        self.image_fx.kill()
        self.image_display.kill()

        self.jack_input.join()
        self.osc_server.join()

    def on_keyboard_interrupt(self, sig, _):
        print("KeyboardInterrupt (ID: {}) has been caught. Cleaning up...".format(sig))
        self.exit_handler()
        exit(0)


def main():
    app = App()
    app.run()


if __name__ == '__main__':
    main()
