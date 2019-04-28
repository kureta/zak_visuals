import queue
import threading

import cv2
import essentia
import essentia.standard as es
import jack
import numpy as np


from berlin.pg_gan.model import Generator
import torch

G: Generator = torch.load('/home/kureta/Documents/repos/berlin/saves/zak1/models/Gs_nch-4_epoch-198.pth').cuda()

########
x = es.MonoLoader(filename='/home/kureta/Music/misc/untitled.wav')()
window = es.Windowing(type='hann')
spectrum = es.Spectrum(size=2048)
specs = []
for frame in es.FrameGenerator(x, frameSize=2048, hopSize=1920):
    specs.append(spectrum(window(frame)))
specs = essentia.array(specs)
mean, std = np.mean(specs), np.std(specs)
########

clientname = 'Zak'
client = jack.Client(clientname)

if client.status.server_started:
    print('JACK server started')
if client.status.name_not_unique:
    print('unique name {0!r} assigned'.format(client.name))

event = threading.Event()

size = 10
specs = np.zeros((size, 1025))
specs_queue = queue.Queue()


def specs_manager():
    global specs
    idx = 0
    while True:
        s = specs_queue.get()
        specs[idx % size] = s
        idx += 1
        specs_queue.task_done()


def launch_daemon(manager):
    t = threading.Thread(target=manager)
    t.daemon = True
    t.start()
    del t


launch_daemon(specs_manager)


@client.set_process_callback
def process(frames):
    assert frames == client.blocksize
    for i in client.inports:
        buffer = i.get_array().astype('float32')
        specc = (spectrum(window(buffer)) - mean) / std
        specs_queue.put(specc)
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

    print('Press Ctrl+C to stop')
    try:
        while True:
            event.wait()
            # image = G(sp)
            features = (specs.mean(axis=0)[:128] - mean) / std
            features = torch.from_numpy(features.astype('float32'))
            features.unsqueeze_(0).unsqueeze_(2).unsqueeze_(3)
            features = features.cuda()
            with torch.no_grad():
                image = G(features)
            image.squeeze_(0).squeeze_(0)
            image = (image + 1) / 2
            image = image.cpu().numpy()

            cv2.imshow('frame', cv2.resize(image, (1920, 1080)))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            event.clear()
    except KeyboardInterrupt:
        print('\nInterrupted by user')
