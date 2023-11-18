import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torchvision
from torch.utils.data import DataLoader
from skimage.feature import peak_local_max
from sklearn.model_selection import train_test_split

from pose_hrnet.models.pose_hrnet import PoseHighResolutionNet
from pose_hrnet.config.default import _C as cfg
from dataset import MPIIDataset

dataset = MPIIDataset()
dataset_size = len(dataset)
train_size = int(dataset_size*0.8)
valid_size = dataset_size - train_size
train_dataset, valid_dataset = train_test_split(dataset, test_size=0.2, shuffle=False)

print("dataset size: {}".format(dataset_size))
print("train size: {}".format(len(train_dataset)))
print("valid size: {}".format(len(valid_dataset)))

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=False, num_workers=0)
valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False, num_workers=0)

best_model = PoseHighResolutionNet(cfg)
best_model.load_state_dict(torch.load('E:/exp/HRNet/epoch20_t_0.00024_v_0.00024.pth'))
best_model = best_model.cuda()
best_model.eval()

with torch.no_grad():
    for step, (images, gt) in enumerate(train_loader):
        
        images = images.reshape(-1, 1, images.shape[1], images.shape[2]).cuda()
        
        predictions = best_model(images)
        
        images = images.cpu()
        predictions = predictions.cpu()
        
        for idx in range(len(images)):
            image = np.array(images[idx][0])
            gt_ = np.array(gt[idx])
            gt_ = np.max(gt_, axis=0)
            prediction = np.array(predictions[idx])
            prediction = np.max(prediction, axis=0)
            
            plt.subplot(1, 3, 1)
            plt.imshow(image)
            plt.subplot(1, 3, 2)
            plt.imshow(gt_)
            plt.subplot(1, 3, 3)
            plt.imshow(prediction)
            plt.show()
        