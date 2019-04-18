import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from berlin.dataset import AutoencoderDataset
from berlin.model.autoencoder import Autoencoder
from berlin.config import Config

BATCH_SIZE = 32
epochs = 5

conf = Config()
dataset = AutoencoderDataset(conf)
data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1, pin_memory=True)

autoencoder = Autoencoder()
autoencoder = autoencoder.cuda()

optimizer = torch.optim.Adam(autoencoder.parameters())
criterion = nn.MSELoss()

for i in range(epochs):
    print(f'epoch {i + 1}')
    for images, audio in data_loader:
        images = images.cuda()
        audio = audio.cuda()

        optimizer.zero_grad()
        x_hat = autoencoder(images, audio)
        loss = criterion(x_hat, images)
        loss.backward()
