import torch.nn as nn

from berlin.model.decoder import Decoder
from berlin.model.encoder import Encoder


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, image_audio):
        image, audio = image_audio
        z = self.encoder(image, audio)
        y = self.decoder(z, audio)

        return y
