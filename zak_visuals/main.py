import ctypes

from torch import multiprocessing as mp

from zak_visuals.nodes import AudioProcessor, AlternativeGenerator, ImageFX, ImageDisplay
from zak_visuals.nodes import JACKInput, OSCServer
from zak_visuals.nodes.base_nodes import Edge


class App:
    def __init__(self):
        mp.set_start_method('spawn', force=True)
        self.exit = mp.Event()

        self.rgb_intensity = mp.Value('f', 1)
        self.rgb_intensity.value = 0.
        self.noise_scale = mp.Value('f', 1)
        self.noise_scale.value = 1.

        self.osc_server = OSCServer(self.exit, rgb_intensity=self.rgb_intensity, noise_scale=self.noise_scale)

        self.buffer = mp.Array(ctypes.c_float, 2048)
        self.cqt = Edge()
        self.image = Edge()
        self.imfx = Edge()

        self.jack_input = JACKInput(outgoing=self.buffer)
        self.audio_processor = AudioProcessor(incoming=self.buffer, outgoing=self.cqt)
        # self.image_generator = ImageGenerator(incoming=self.cqt, outgoing=self.image)
        self.image_generator = AlternativeGenerator(incoming=self.cqt, outgoing=self.image,
                                                    noise_scale=self.noise_scale)
        self.image_fx = ImageFX(incoming=self.image, outgoing=self.imfx, rgb_intensity=self.rgb_intensity)
        self.image_display = ImageDisplay(incoming=self.imfx, exit_event=self.exit)

    def run(self):
        self.osc_server.start()
        self.jack_input.start()
        self.audio_processor.start()
        self.image_generator.start()
        self.image_fx.start()
        self.image_display.start()
        self.exit.wait()
        self.exit_handler()

    def exit_handler(self):
        self.jack_input.stop()
        self.audio_processor.stop()
        self.image_generator.stop()
        self.image_fx.stop()
        self.image_display.stop()
        self.osc_server.stop()


def main():
    app = App()
    app.run()


if __name__ == '__main__':
    main()
