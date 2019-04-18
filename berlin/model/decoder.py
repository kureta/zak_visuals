import torch
import torch.nn as nn
from torch.nn import functional as F
from functools import partial


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(512 + 56, 1024)
        self.fc2 = nn.Linear(1024, 2048)
        self.fc3 = nn.Linear(2048, 4096)

        self.conv_1x1 = nn.Conv2d(in_channels=64,
                                  out_channels=128,
                                  kernel_size=1)
        self.conv1 = nn.Conv2d(in_channels=128,
                               out_channels=128,
                               kernel_size=3,
                               padding=1)
        self.conv2 = nn.Conv2d(in_channels=128,
                               out_channels=128,
                               kernel_size=3,
                               padding=1)
        self.conv3 = nn.Conv2d(in_channels=128,
                               out_channels=128,
                               kernel_size=3,
                               padding=1)
        self.conv4 = nn.Conv2d(in_channels=128,
                               out_channels=64,
                               kernel_size=3,
                               padding=1)
        self.conv5 = nn.Conv2d(in_channels=64,
                               out_channels=3,
                               kernel_size=3,
                               padding=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.interpolate = partial(F.interpolate, scale_factor=2, mode='bilinear')

    def forward(self, z, audio):
        y = torch.cat([z, audio], dim=1)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.relu(y)
        y = self.fc3(y)
        y = self.relu(y)

        y = y.view(y.shape[0], 64, 8, 8)

        y = self.conv_1x1(y)

        y = self.interpolate(y)
        y = self.conv1(y)
        y = self.relu(y)
        y = self.interpolate(y)
        y = self.conv2(y)
        y = self.relu(y)
        y = self.interpolate(y)
        y = self.conv3(y)
        y = self.relu(y)
        y = self.interpolate(y)
        y = self.conv4(y)
        y = self.relu(y)
        y = self.interpolate(y)
        y = self.conv5(y)
        y = self.sigmoid(y)

        return y
