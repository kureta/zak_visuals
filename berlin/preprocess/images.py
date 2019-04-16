import cv2

from berlin.config import Config
from berlin.preprocess.utils import recursive_file_paths, do_multiprocess


def calculate_image_features(config):
    image_files = recursive_file_paths(config.image_dir, config.image_extensions)
    # images = do_multiprocess(cv2.imread, image_files)
    return image_files


if __name__ == '__main__':
    print(len(calculate_image_features(Config())))
