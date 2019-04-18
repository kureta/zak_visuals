from berlin.dataset import BaseDataset


class AudioDataset(BaseDataset):
    def __init__(self, path):
        super(AudioDataset, self).__init__(path)
