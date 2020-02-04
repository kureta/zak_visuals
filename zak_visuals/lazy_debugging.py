import ctypes
import logging
import signal

from torch import multiprocessing as mp

from zak_visuals.nodes import OSCServer, NoiseGenerator

logger = mp.log_to_stderr()
logger.setLevel(logging.DEBUG)


def on_keyboard_interrupt(sig, _):
    print("KeyboardInterrupt (ID: {}) has been caught. Cleaning up...".format(sig))
    noise_generator.kill()
    osc_server.join()
    exit(0)


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    app_exit = mp.Event()

    rgb = mp.Value(ctypes.c_float)
    stft_scale = mp.Value(ctypes.c_float)
    animate_noise = mp.Value(ctypes.c_float)
    params = {'rbg': rgb, 'stft_scale': stft_scale, 'animate_noise': animate_noise}
    rgb.value, stft_scale.value, animate_noise.value = 0., 0., 0.,

    noise = mp.Queue(maxsize=1)

    osc_server = OSCServer(app_exit, params=params)
    noise_generator = NoiseGenerator(outgoing=noise, params=params)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    osc_server.start()
    noise_generator.start()
    signal.signal(signal.SIGINT, on_keyboard_interrupt)

    idx = 0
    while not app_exit.is_set():
        x = noise.get()
        idx += 1

    on_keyboard_interrupt('sinyalsiz', 'lolo')
