import os
import cv2
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset
import torchvision

class MPIIDataset(Dataset):
    def __init__(self, transform=None):
        self.images_list = []
        self.gt_list = []
        self.transform = transform
        
        # load images
        image_dir = 'D:/python_virtual/newvenv/data/images/'
        image_name_list = os.listdir(image_dir)
        print(len(image_name_list))
        for name in image_name_list:
            image = cv2.imread(image_dir + name, 0)
            
            # normalization
            image = image / 255
            
            

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        image = self.images_list[idx]
        gt = self.gt_list[idx]
        
        return image, gt

data = MPIIDataset()