import ctypes
import logging
import signal

from torch import multiprocessing as mp

from zak_visuals.nodes import AudioProcessor, InteropDisplay, NoiseGenerator, StyleGAN2
from zak_visuals.nodes import JACKInput, OSCServer

logger = mp.log_to_stderr()
logger.setLevel(logging.ERROR)


# TODO: Test Pipe instead of Queue
# TODO: use longer queues, buffer inputs, generate batch images
class App:
    def __init__(self):
        mp.set_start_method('spawn', force=True)
        self.exit = mp.Event()

        pause_stylegan = mp.Event()
        pause_audio = mp.Event()
        pause_noise = mp.Event()

        rgb = mp.Value(ctypes.c_float, lock=False)
        stft_scale = mp.Value(ctypes.c_float, lock=False)
        animate_noise = mp.Value(ctypes.c_float, lock=False)
        randomize_label = mp.Value(ctypes.c_float, lock=False)
        label_group = mp.Value(ctypes.c_uint8, lock=False)
        label_speed = mp.Value(ctypes.c_float, lock=False)
        noise_speed = mp.Value(ctypes.c_float, lock=False)
        noise_std = mp.Value(ctypes.c_float, lock=False)
        rms_influence = mp.Value(ctypes.c_float, lock=False)
        params = {
            'rgb': rgb,
            'stft_scale': stft_scale,
            'animate_noise': animate_noise,
            'randomize_label': randomize_label,
            'label_group': label_group,
            'label_speed': label_speed,
            'noise_speed': noise_speed,
            'noise_std': noise_std,
            'pause_gans': [pause_stylegan],
            'pause_all': [pause_audio, pause_noise, pause_stylegan],
            'rms_influence': rms_influence
        }
        rgb.value, stft_scale.value, animate_noise.value, randomize_label.value = 0., 0., 0., 0.
        label_speed.value, noise_speed.value, noise_std.value, rms_influence.value = 0., 0., 0., 0.
        label_group.value = 0

        buffer = mp.Array(ctypes.c_float, 2048, lock=False)
        stft = mp.Array(ctypes.c_float, 128, lock=False)
        rms = mp.Value(ctypes.c_float, lock=False)
        noise = mp.Queue(maxsize=1)
        image = mp.Queue(maxsize=1)

        self.osc_server = OSCServer(self.exit, params=params)
        self.jack_input = JACKInput(outgoing=buffer)
        self.audio_processor = AudioProcessor(incoming=buffer, outgoing=stft, rms=rms, pause_event=pause_audio)
        self.stylegan = StyleGAN2(pause_event=pause_stylegan, noise=noise, stft_in=stft, outgoing=image,
                                  params=params)
        self.noise_generator = NoiseGenerator(outgoing=noise, params=params, pause_event=pause_noise)
        self.image_display = InteropDisplay(incoming=image, exit_app=self.exit)

    def run(self):
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        self.osc_server.start()
        self.jack_input.start()

        self.audio_processor.start()
        self.noise_generator.start()
        self.stylegan.start()
        self.image_display.start()

        signal.signal(signal.SIGINT, self.on_keyboard_interrupt)

        self.exit.wait()
        self.exit_handler()

    def exit_handler(self):
        self.audio_processor.kill()
        self.noise_generator.kill()
        self.stylegan.kill()
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
