import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

class WineDataset(Dataset):
    
    def __init__(self):
        # data loading
        xy = np.loadtxt('./data/wine/wine.csv', delimiter=",", dtype=np.float32, skiprows=1) # skip header
        self.x = torch.from_numpy(xy[:, 1:]) # all observations, all attributes except the first column (the type of wine) (dimensions = n_samples, n_attributes)
        self.y = torch.from_numpy(xy[:, [0]]) # all observations, only the first column (the type of wine) (dimensions = n_samples, 1)
        self.n_samples = xy.shape[0]
        
    def __getitem__(self, index):
        # dataset[0] -> returns a tuple for the attributes and the value
        return self.x[index], self.y[index]
        
    def __len__(self):
        # len(dataset)
        return self.n_samples
    
dataset = WineDataset()
batch_size = 4
dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

data_iter = iter(dataloader)
data = next(data_iter) # a batch of size 4 => 4 input tensors and their respective 4 outputs
features, labels = data

# training loop
num_epochs = 2
total_samples = len(dataset)
n_batches = math.ceil(total_samples / batch_size) # how many batches?

for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(dataloader):
        # forward, backward, update
        if (i + 1) % 5 == 0:
            print(f'epoch {epoch+1}/{num_epochs}, step {i+1}/{n_batches}, inputs {inputs.shape}')