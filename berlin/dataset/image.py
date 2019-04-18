from berlin.dataset import BaseDataset


class ImageDataset(BaseDataset):
    def __init__(self, path):
        super(ImageDataset, self).__init__(path)
