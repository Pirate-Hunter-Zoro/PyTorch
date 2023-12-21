"""
Documentation if you REALLY care to read it: 
https://pytorch.org/vision/0.9/transforms.html

Apply to PIL images, tensors, ndarrays, or custom data during creation of the Dataset

On Images
---------
CenterCrop,, Grayscale, Pad, RandomAffine, RandomCrop, RandomHorizontalFlip, RandomRotation, Resize, Scale

On Tensors
----------
LinearTransformation, Normalize, RandomErasing

Conversion
----------
ToPILImage: from tensor or ndarray
ToTensor: from numpy.ndarray or PILImage

Generic
-------
Use Lambda

Custom
------
Write own class

Compose multiple Transforms
---------------------------
composed = transforms.Compose([Rescale(256),
                            RandomCrop(224)])

torchvision.transforms.ReScale(256)
torchvision.transforms.ToTensor()
"""
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np


class WineDataset(Dataset):
    def __init__(self, transform=None):
        xy = np.loadtxt(
            "./data/wine/wine.csv", delimiter=",", dtype=np.float32, skiprows=1
        )  # skip the first row because that's just the attribute labels
        self.n_samples = xy.shape[0]

        # note that we do not convert to tensor here - these are numpy arrays
        self.x = xy[
            :, 1:
        ]  # all observations/rows, skip the first column which is the y value (wine classification)
        self.y = xy[
            :, [0]
        ]  # all observations/rows, ONLY the first column which is the y value (wine classification)

        self.transform = transform

    def __getitem__(self, index):  # returns a tuple
        sample = self.x[index], self.y[index]

        if self.transform:  # not None
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return self.n_samples


class ToTensor:
    def __call__(self, sample):  # still returning a tuple
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)


class MulTransform:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, sample):
        inputs, target = sample
        inputs *= self.factor
        return inputs, target


dataset = WineDataset(transform=ToTensor())
first_data = dataset[0]
features, labels = first_data
print(features)
print(type(features), type(labels))

composed = transforms.Compose([ToTensor(), MulTransform(4)])
dataset = WineDataset(transform=composed)
first_data = dataset[0]
features, labels = first_data
print(features)
print(type(features), type(labels))
