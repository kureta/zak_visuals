import queue
import threading

import cv2
import essentia.standard as es
import jack
import numpy as np
import torch
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

from berlin.pg_gan.model import Generator
from berlin.ui.main import Ui_MainWindow

G: Generator = torch.load('/home/kureta/Documents/repos/berlin/saves/zak1/models/Gs_nch-4_epoch-198.pth').cuda()
mean = np.load('mel_mean.npy')
std = np.load('mel_std.npy')

event = threading.Event()

smooth = 30
specs = np.zeros((smooth, 128))
amps = np.zeros(smooth)
specs_queue = queue.Queue()


def specs_manager():
    global specs
    idx = 0
    while True:
        s, a = specs_queue.get()
        specs[idx % smooth] = s
        amps[idx % smooth] = a
        idx += 1
        specs_queue.task_done()


def launch_daemon(manager):
    t = threading.Thread(target=manager)
    t.daemon = True
    t.start()
    del t


def hypersphere(z, radius=1.):
    return (z / np.linalg.norm(z, ord=2)) * radius


client = jack.Client('Zak')

if client.status.server_started:
    print('JACK server started')
if client.status.name_not_unique:
    print('unique name {0!r} assigned'.format(client.name))

# create two port pairs
for number in 1, 2:
    client.inports.register('input_{0}'.format(number))


@client.set_process_callback
def process(frames):
    assert frames == client.blocksize
    lolo = []
    amplis = []
    for i in client.inports:
        buffer = i.get_array().astype('float32')
        sp = spectrum(window(buffer))
        lolo.append((mel_bands(sp) - mean) / std)

        amplis.append(np.sqrt((buffer * buffer).mean()))

    amp = sum(amplis)
    specs_queue.put((np.concatenate(lolo), amp))
    event.set()


@client.set_shutdown_callback
def shutdown(status, reason):
    print('JACK shutdown!')
    print('status:', status)
    print('reason:', reason)
    event.set()


window = es.Windowing(type='hann', size=2048)
spectrum = es.Spectrum(size=2048)
mel_bands = es.MelBands(inputSize=1025,
                        highFrequencyBound=12000,
                        lowFrequencyBound=46.875,
                        numberBands=64, sampleRate=48000)


class RenderZak(QThread):
    def __init__(self):
        QThread.__init__(self)

        self.gain = 0.
        self.curve = 0.

    def run(self):
        with client:
            cv2.namedWindow('frame', cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty('frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

            print('Press Ctrl+C to stop')
            try:
                while True:
                    event.wait()
                    features = specs[:smooth].mean(axis=0)
                    amplis = amps[:smooth].mean()
                    amplis = (amplis ** self.curve * self.gain)

                    features = hypersphere(features)

                    features = torch.from_numpy(features.astype('float32'))
                    features.unsqueeze_(0).unsqueeze_(2).unsqueeze_(3)
                    features = features.cuda()
                    with torch.no_grad():
                        image = G(features)
                    image.squeeze_(0).squeeze_(0)
                    image = (image + 1) / 2
                    image = image.cpu().numpy()

                    frame = cv2.resize(image, (1920, 1080)) * amplis
                    cv2.imshow('frame', frame)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        cv2.destroyAllWindows()
                        break

                    event.clear()
            except KeyboardInterrupt:
                print('\nInterrupted by user')


def main():
    import sys
    app = QApplication(sys.argv)
    main_window = QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(main_window)

    # other code
    launch_daemon(specs_manager)
    p = RenderZak()
    ui.run_button.clicked.connect(p.run)

    def set_values():
        global smooth
        p.gain = (ui.gain_slider.value() / 1000) * 5.
        p.curve = (ui.curve_slider.value() / 1000)
        smooth = ui.smooth_slider.value()

        ui.curve_label.setText(f'{p.curve}')
        ui.gain_label.setText(f'{p.gain}')
        ui.smooth_label.setText(f'{smooth}')

    ui.gain_slider.valueChanged.connect(set_values)
    ui.curve_slider.valueChanged.connect(set_values)
    ui.smooth_slider.valueChanged.connect(set_values)

    main_window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
