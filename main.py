import os
import cv2
import sys
import numpy as np

import torch
import torch.nn as nn
import torchvision
from torch import optim
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split

from pose_hrnet.models.pose_hrnet import PoseHighResolutionNet
from pose_hrnet.config.default import _C as cfg
from dataset import MPIIDataset

epoch_size = 100

dataset = MPIIDataset()
dataset_size = len(dataset)
train_size = int(dataset_size*0.8)
valid_size = dataset_size - train_size
train_dataset, valid_dataset = train_test_split(dataset, test_size=0.2, shuffle=False)
# train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

print("dataset size: {}".format(dataset_size))
print("train size: {}".format(len(train_dataset)))
print("valid size: {}".format(len(valid_dataset)))

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0)

criterion = nn.MSELoss(reduction='mean')

model = PoseHighResolutionNet(cfg)

load_model = False
start_epoch = 0

if load_model:
    model.load_state_dict(torch.load('E:/exp/HRNet/epoch3_t_0.00477_v_0.00462.pth'))
    start_epoch = 4
    
model.to(torch.device('cuda'))

optimizer = optim.Adam(model.parameters(), lr=0.0001)

for epoch in range(start_epoch, epoch_size):
    train_loss = 0
    valid_loss = 0
    
    model.train()
    for step, (images, gt) in enumerate(train_loader):

        images = images.reshape(-1, 1, images.shape[1], images.shape[2]).cuda()
        gt = gt.cuda()
        
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
            gt = gt.cuda()
            
            predictions = model(images)
            
            loss = criterion(predictions, gt)
            
            valid_loss += loss.item()
            sys.stdout.write('\r')
            sys.stdout.write("step: %d/%d  valid_loss: %.5f" % (step + 1, len(valid_loader), valid_loss/(step + 1)))
            sys.stdout.flush()
        
    valid_loss = valid_loss / len(valid_loader)
    print("\nepoch: {}  valid_loss: {:.5f}\n".format(epoch, valid_loss))
    
    path = 'E:/exp/HRNet/epoch{}_t_{:.5f}_v_{:.5f}.pth'.format(epoch, train_loss, valid_loss)
    torch.save(model.state_dict(), path)