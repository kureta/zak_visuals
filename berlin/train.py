import torch
import torch.nn as nn
from ignite.engine import create_supervised_trainer, Events
from torch.utils.data import DataLoader

from berlin.config import Config
from berlin.dataset import AutoencoderDataset
from berlin.model.autoencoder import Autoencoder

BATCH_SIZE = 32
epochs = 5

conf = Config()
dataset = AutoencoderDataset(conf)
data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1, pin_memory=True)

autoencoder = Autoencoder()

optimizer = torch.optim.Adam(autoencoder.parameters())
criterion = nn.MSELoss()

trainer = create_supervised_trainer(autoencoder, optimizer, criterion, 'cuda')


@trainer.on(Events.ITERATION_COMPLETED)
def log_training_loss(tr):
    print("Epoch[{}] Loss: {:.2f}".format(tr.state.epoch, tr.state.output))


trainer.run(data_loader, max_epochs=5)
