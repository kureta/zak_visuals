import cv2
import numpy as np

from berlin.config import Config
from berlin.preprocess.utils import recursive_file_paths, do_multiprocess


def load_image(image_file):
    image = cv2.imread(image_file).astype('float16')
    return image / 255.


def calculate_image_features(config):
    image_files = recursive_file_paths(config.image_dir, config.image_extensions)
    images = do_multiprocess(load_image, image_files)
    return np.stack(images)


def load_saved():
    return np.load('data/pickles/image_features.npy')


def main():
    image_features = calculate_image_features(Config())
    np.save('data/pickles/image_features.npy', image_features)


if __name__ == '__main__':
    main()
