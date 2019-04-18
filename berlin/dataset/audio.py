from berlin.dataset import BaseDataset


class AudioDataset(BaseDataset):
    def __init__(self, path):
        super(AudioDataset, self).__init__(path)

        self.data = self.data.as_strided((self.data.shape[0], 256, 256, self.data.shape[1]),
                                         (self.data.stride()[0], 0, 0, self.data.stride()[1]))
