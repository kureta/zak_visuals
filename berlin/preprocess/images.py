import cv2
import numpy as np
import torch

from berlin.config import Config
from berlin.preprocess.utils import recursive_file_paths, do_multiprocess


def load_image(image_file):
    image = cv2.imread(image_file).astype('float32')
    return image / 255.


def calculate_image_features(config):
    image_files = recursive_file_paths(config.image_dir, config.image_extensions)
    images = do_multiprocess(load_image, image_files)
    return np.stack(images)


def images_to_tensor(save=False):
    image_features = torch.from_numpy(calculate_image_features(Config()))
    image_features = image_features.permute(0, 3, 1, 2)
    if save:
        torch.save(image_features, 'data/pickles/images.torch')

    return image_features


if __name__ == '__main__':
    images_to_tensor(save=True)
