from functools import partial

import torch.nn as nn
from torch.nn import functional as F


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=64,
                               kernel_size=3,
                               padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3,
                                     stride=2,
                                     padding=1)
        self.conv2 = nn.Conv2d(in_channels=64,
                               out_channels=128,
                               kernel_size=3,
                               padding=1)

        self.relu = nn.LeakyReLU()
        self.tanh = nn.Tanh()

    def forward(self, image, audio):
        z = self.conv1(image)
        z = self.relu(z)
        z = self.maxpool1(z)
        z = self.conv2(z)
        z = self.tanh(z)

        return z


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=128,
                               out_channels=64,
                               kernel_size=3,
                               padding=1)
        self.conv2 = nn.Conv2d(in_channels=64,
                               out_channels=3,
                               kernel_size=3,
                               padding=1)
        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.interpolate = partial(F.interpolate, scale_factor=2, mode='bilinear')

    def forward(self, z, audio):
        image = self.conv1(z)
        image = self.relu(image)
        image = self.interpolate(image)
        image = self.conv2(image)
        image = self.sigmoid(image)

        return image


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
