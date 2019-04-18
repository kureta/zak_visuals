import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=59,
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
        self.maxpool2 = nn.MaxPool2d(kernel_size=3,
                                     stride=2,
                                     padding=1)
        self.conv3 = nn.Conv2d(in_channels=128,
                               out_channels=128,
                               kernel_size=3,
                               padding=1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3,
                                     stride=2,
                                     padding=1)
        self.conv4 = nn.Conv2d(in_channels=128,
                               out_channels=128,
                               kernel_size=3,
                               padding=1)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3,
                                     stride=2,
                                     padding=1)
        self.conv5 = nn.Conv2d(in_channels=128,
                               out_channels=128,
                               kernel_size=3,
                               padding=1)
        self.maxpool5 = nn.MaxPool2d(kernel_size=3,
                                     stride=2,
                                     padding=1)
        self.conv_1x1 = nn.Conv2d(in_channels=128,
                                  out_channels=64,
                                  kernel_size=1)

        current_size = 8 * 8 * 64
        self.fc1 = nn.Linear(current_size, current_size // 2)
        current_size //= 2
        self.fc2 = nn.Linear(current_size, current_size // 2)
        current_size //= 2
        self.fc3 = nn.Linear(current_size, current_size // 2)
        self.z_dims = current_size // 2

        self.relu = nn.ReLU()

    def forward(self, image, audio):
        audio = audio.as_strided((audio.shape[0], audio.shape[1], 256, 256),
                                 (audio.stride()[0], audio.stride()[1], 0, 0))

        x = torch.cat([image, audio], dim=1)
        z = self.conv1(x)
        z = self.relu(z)
        z = self.maxpool1(z)
        z = self.conv2(z)
        z = self.relu(z)
        z = self.maxpool2(z)
        z = self.conv3(z)
        z = self.relu(z)
        z = self.maxpool3(z)
        z = self.conv4(z)
        z = self.relu(z)
        z = self.maxpool4(z)
        z = self.conv5(z)
        z = self.relu(z)
        z = self.maxpool5(z)

        z = self.conv_1x1(z)

        z = z.view(z.shape[0], -1)

        z = self.fc1(z)
        z = self.relu(z)
        z = self.fc2(z)
        z = self.relu(z)
        z = self.fc3(z)

        return z
