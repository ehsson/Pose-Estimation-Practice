import os
import cv2
import sys
import numpy

import torch
import torchvision
from torch.utils.data import DataLoader

from dataset import MPIIDataset

epoch_size = 100

dataset = MPIIDataset()
train_loader = DataLoader(dataset, batchsize=32, shuffle=True)
valid_loader = DataLoader(dataset, batchsize=32, shuffle=True)

for epoch in range(1, epoch_size):
    for step in range(1, len(train_loader)):
        pass
    
    for step in range(1, len(valid_loader)):
        pass