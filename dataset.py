import os
import cv2
import json
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
        dir = 'D:/data/mpii_data/'
        image_dir = dir + 'mpii_human_pose_v1/images/'
        annotation_dir = dir
        image_name_list = []

        with open(annotation_dir + 'mpii_annotations.json', 'r') as f:
            json_data = json.load(f)

            for i in range(len(json_data)):
                if json_data[i]['img_width'] == 1280 and json_data[i]['img_height'] == 720:
                    image_name_list.append(json_data[i]['img_paths'])
            
            

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        image = self.images_list[idx]
        gt = self.gt_list[idx]
        
        return image, gt

data = MPIIDataset()