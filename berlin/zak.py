import queue
import threading
import os

import cv2
import essentia
import essentia.standard as es
import jack
import numpy as np

from berlin.pg_gan.model import Generator
from berlin.dataset.audio import VIDEO_DIR, AUDIO_FILE_NAME
import torch
from torch.utils.data import ConcatDataset

G: Generator = torch.load('/home/kureta/Documents/repos/berlin/saves/zak1/models/Gs_nch-4_epoch-198.pth').cuda()

########
x = es.MonoLoader(filename='/home/kureta/Music/misc/untitled.wav')()
window = es.Windowing(type='hann', size=2048)
spectrum = es.Spectrum(size=2048)
mel_bands = es.MelBands(inputSize=1025,
                        highFrequencyBound=12000,
                        lowFrequencyBound=46.875,
                        numberBands=64, sampleRate=48000)
loudness = es.Loudness()


def spect(video_idx):
    audio_path = os.path.join(VIDEO_DIR, AUDIO_FILE_NAME.format(video_idx + 1))

    easy_loader = es.EasyLoader(filename=audio_path,
                                replayGain=0,
                                sampleRate=48000)

    audio = np.pad(easy_loader(), 64, 'reflect')
    frame_generator = es.FrameGenerator(audio, frameSize=2048, hopSize=48000 // 25)

    mels = []
    for frame in frame_generator:
        windowed = window(frame)
        spec = spectrum(windowed)
        mel = mel_bands(spec)
        mels.append(mel)

    return np.concatenate(np.expand_dims(mels, 0), 0)


list_mels = [spect(idx) for idx in range(9)]
mels = np.concatenate(list_mels, 0)

mean = mels.mean(0)
std = mels.std(0)
########


clientname = 'Zak'
client = jack.Client(clientname)

if client.status.server_started:
    print('JACK server started')
if client.status.name_not_unique:
    print('unique name {0!r} assigned'.format(client.name))

event = threading.Event()

size = 16
specs = np.zeros((size, 128))
amps = np.zeros(size)
specs_queue = queue.Queue()


def specs_manager():
    global specs
    idx = 0
    while True:
        s, a = specs_queue.get()
        specs[idx % size] = s
        amps[idx % size] = a
        idx += 1
        specs_queue.task_done()


def launch_daemon(manager):
    t = threading.Thread(target=manager)
    t.daemon = True
    t.start()
    del t


launch_daemon(specs_manager)


def hypersphere(z, radius=1):
    return z * radius / np.linalg.norm(z, ord=2)


@client.set_process_callback
def process(frames):
    assert frames == client.blocksize
    lolo = []
    amps = []
    for i in client.inports:
        buffer = i.get_array().astype('float32')
        sp = spectrum(window(buffer))
        lolo.append((mel_bands(sp) - mean) / std)

        amps.append(np.sqrt((buffer * buffer).mean()))

    for midis in client.midi_inports[0].incoming_midi_events():
        print(midis)

    amp = sum(amps)
    specs_queue.put((np.concatenate(lolo), amp))
    event.set()


@client.set_shutdown_callback
def shutdown(status, reason):
    print('JACK shutdown!')
    print('status:', status)
    print('reason:', reason)
    event.set()


# create two port pairs
for number in 1, 2:
    client.inports.register('input_{0}'.format(number))

with client:
    capture = client.get_ports(is_physical=True, is_output=True)
    if not capture:
        raise RuntimeError('No physical capture ports')

    for src, dest in zip(capture, client.inports):
        client.connect(src, dest)

    out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), 25.0, (1920, 1080))
    cv2.namedWindow('frame', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print('Press Ctrl+C to stop')
    try:
        while True:
            event.wait()
            # image = G(sp)
            features = specs.mean(axis=0)
            amplis = amps.mean()
            amplis = (amplis * 3)
            print(amplis)

            features = hypersphere(features, amplis)

            features = torch.from_numpy(features.astype('float32'))
            features.unsqueeze_(0).unsqueeze_(2).unsqueeze_(3)
            features = features.cuda()
            with torch.no_grad():
                image = G(features)
            image.squeeze_(0).squeeze_(0)
            image = (image + 1) / 2
            image = image.cpu().numpy()

            frame = cv2.resize(image, (1920, 1080)) * amplis
            out.write(np.uint8(frame))
            cv2.imshow('frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                out.release()
                cv2.destroyAllWindows()
                break

            event.clear()
    except KeyboardInterrupt:
        print('\nInterrupted by user')
