import pickle
from dataclasses import dataclass

import essentia
import essentia.standard as es
import numpy as np

from berlin.config import Config
from berlin.preprocess.utils import recursive_file_paths


@dataclass
class Features:
    mel_bands: np.array
    mfcc: np.array
    loudness: np.array
    flatness: np.array
    flux: np.array


def calculate_audio_features(config):
    # TODO: multiple audio files
    audio_path = recursive_file_paths(config.audio_dir, config.audio_extensions)[0]
    easy_loader = es.EasyLoader(filename=audio_path,
                                replayGain=0,
                                sampleRate=config.sample_rate)
    audio = easy_loader()
    frame_gen = es.FrameGenerator(audio,
                                  frameSize=config.frame_size,
                                  hopSize=config.hop_size,
                                  startFromZero=True,
                                  validFrameThresholdRatio=1)
    window_fn = es.Windowing(size=config.frame_size,
                             type='hann')
    spectrum_fn = es.Spectrum(size=config.frame_size)
    mfcc_fn = es.MFCC(inputSize=config.spectrum_size,
                      sampleRate=config.sample_rate)
    loudness_fn = es.Loudness()
    flux_fn = es.Flux()
    flatness_fn = es.Flatness()

    pool = essentia.Pool()

    for frame in frame_gen:
        windowed_frame = window_fn(frame)
        spectrum = spectrum_fn(windowed_frame)
        mel_bands, mfcc = mfcc_fn(spectrum)
        loudness = loudness_fn(frame)
        flux = flux_fn(spectrum)
        flatness = flatness_fn(spectrum)

        pool.add('mel_bands', mel_bands)
        pool.add('mfcc', mfcc)
        pool.add('loudness', loudness)
        pool.add('flux', flux)
        pool.add('flatness', flatness)

    features = pool_to_features(pool, config)

    return features


def pool_to_features(pool, config):
    result = dict()
    for key in pool.descriptorNames():
        values = pool[key]
        array = essentia.array(values)
        array = array[config.frames_start_index:config.frames_end_index]
        array = (array - np.mean(array)) / np.std(array)
        result[key] = array

    return Features(**result)


def load_saved():
    with open('data/pickles/audio_features.pickle', 'rb') as f:
        # The protocol version used is detected automatically, so we do not
        # have to specify it.
        audio_features = pickle.load(f)

    return audio_features


def main():
    audio_features = calculate_audio_features(Config())
    with open('data/pickles/audio_features.pickle', 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(audio_features, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
