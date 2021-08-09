import pandas as pd
from torch.utils.data import Dataset
import torch
from torch.utils.data.dataloader import DataLoader

from dataSetCreator2 import FeatureDataset

import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam

# initialize dataset
data = FeatureDataset('5_Saccades_lucy1.csv', 2)

# load dataset
data_train = DataLoader(dataset = data, batch_size = 10, shuffle = False)

# define Net
class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(5, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 7),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = Network()

# define device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

# shift net to device
model.to(device)

# define parameters
learning_rate = 1e-3
batch_size = 64
epochs = 5

# Initialize the loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# define training loop
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# train
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(data_train, model, loss_fn, optimizer)
print("Done!")