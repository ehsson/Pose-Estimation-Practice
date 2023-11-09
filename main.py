import os
import cv2
import sys
import numpy as np

import torch
import torch.nn as nn
import torchvision
from torch import optim
from torch.utils.data import DataLoader, random_split

from pose_hrnet.models.pose_hrnet import PoseHighResolutionNet
from pose_hrnet.config.default import _C as cfg
from dataset import MPIIDataset

epoch_size = 100

dataset = MPIIDataset()
dataset_size = len(dataset)
train_size = int(dataset_size * 0.8)
valid_size = dataset_size - train_size
train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

print("train size: {}".format(train_size))
print("valid_size: {}".format(valid_size))

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
valid_loader = DataLoader(valid_dataset, batch_size=2, shuffle=False, num_workers=0)

criterion = nn.MSELoss(reduce='mean')

model = PoseHighResolutionNet(cfg)
model = model.cuda()

optimizer = optim.Adam(model.parameters(), lr=0.0001)

for epoch in range(1, epoch_size):
    train_loss = 0
    valid_loss = 0
    
    model.train()
    for step, (images, gt) in enumerate(train_loader):
        
        images = images.reshape(-1, 1, images.shape[1], images.shape[2]).cuda()
        gt = gt.reshape(-1, 1, gt.shape[1], gt.shape[2]).cuda()
        
        predictions = model(images)
        
        loss = criterion(predictions, gt)
        
        model.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        sys.stdout.write('\r')
        sys.stdout.write("step: %d/%d  train_loss: %.5f" % (step + 1, len(train_loader), train_loss/(step + 1)))
        sys.stdout.flush()
    
    train_loss = train_loss / len(train_loader)
    print("\nepoch: {}  train_loss: {:.5f}".format(epoch, train_loss))
    
    model.eval()
    with torch.no_grad():
        for step, (images, gt) in enumerate(valid_loader):
            
            images = images.reshape(-1, 1, images.shape[1], images.shape[2]).cuda()
            gt = gt.reshape(-1, 1, gt.shape[1], gt.shape[2]).cuda()
            
            predictions = model(images)
            
            loss = criterion(predictions, gt)
            
            valid_loss += loss.item()
            sys.stdout.write('\r')
            sys.stdout.write("step: %d/%d  valid_loss: %.5f" % (step + 1, len(valid_loader), valid_loss/(step + 1)))
            sys.stdout.flush()
        
    valid_loss = valid_loss / len(valid_loader)
    print("\nepoch: {}  valid_loss: {:.5f}\n".format(epoch, valid_loss))
    
    path = 'E:/exp/HRNet/epoch{}_t_{:.5f}_v_{:.5f}.pth'.format(epoch, train_loss, valid_loss)
    torch.save(model, path)