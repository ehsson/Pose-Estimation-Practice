import os
import cv2
import sys
import numpy as np

import torch
import torch.nn as nn
import torchvision
from torch import optim
from torch.utils.data import DataLoader, random_split

from dataset import MPIIDataset

epoch_size = 100

dataset = MPIIDataset()
dataset_size = len(dataset)
train_size = int(dataset_size * 0.8)
valid_size = int(dataset_size * 0.2)
train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

print("train size: {}".format(train_size))
print("valid_size: {}".format(valid_size))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

criterion = nn.MSELoss(reduce='mean')

model = None

optimizer = optim.Adam(model.parameters(), lr=0.0001)

for epoch in range(1, epoch_size):
    train_loss = 0
    valid_loss = 0
    
    model.train()
    for step, (image, gt) in enumerate(train_loader):
        
        image = image.reshape(-1, 1, image.shape[1], image.shape[2]).cuda()
        gt = gt.reshape(-1, 1, gt.shape[1], gt.shape[2]).cuda()
        
        predictions = model(image)
        
        loss = criterion(predictions, gt)
        
        model.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        sys.stdout.write('\r')
        sys.stdout.write("step: %d/%d  train_loss: %.5f" % (step, len(train_loader), train_loss/step))
        sys.stdout.flush()
    
    train_loss = train_loss / len(train_loader)
    print("\nepoch: {}  train_loss: {:.5f}".format(epoch, train_loss))
    
    model.eval()
    with torch.no_grad():
        for step, (image, gt) in enumerate(valid_loader):
            
            image = image.reshape(-1, 1, image.shape[1], image.shape[2]).cuda()
            gt = gt.reshape(-1, 1, gt.shape[1], gt.shape[2]).cuda()
            
            predictions = model(image)
            
            loss = criterion(predictions, gt)
            
            valid_loss += loss.item()
            sys.stdout.write('\r')
            sys.stdout.write("step: %d/%d  valid_loss: %.5f" % (step, len(valid_loader), valid_loss/step))
            sys.stdout.flush()
        
    valid_loss = valid_loss / len(valid_loader)
    print("\nepoch: {}  valid_loss: {:.5f}\n".format(epoch, valid_loss))