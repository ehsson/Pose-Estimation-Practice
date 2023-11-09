import os
import cv2
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset
import torchvision
from skimage.feature import peak_local_max

def generate_gaussian(t, x, y, sigma=10):
    """
    Generates a 2D Gaussian point at location x,y in tensor t.

    x should be in range (-1, 1) to match the output of fastai's PointScaler.

    sigma is the standard deviation of the generated 2D Gaussian.
    """
    h,w = t.shape

    # Heatmap pixel per output pixel
    # mu_x = int(0.5 * (x + 1.) * w)
    # mu_y = int(0.5 * (y + 1.) * h)
    mu_x = x
    mu_y = y

    tmp_size = sigma * 3

    # Top-left
    x1,y1 = int(mu_x - tmp_size), int(mu_y - tmp_size)

    # Bottom right
    x2, y2 = int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)
    if x1 >= w or y1 >= h or x2 < 0 or y2 < 0:
        return t

    size = 2 * tmp_size + 1
    tx = np.arange(0, size, 1, np.float32)
    ty = tx[:, np.newaxis]
    x0 = y0 = size // 2
    _gaussians = {}
    # The gaussian is not normalized, we want the center value to equal 1
    g = _gaussians[sigma] if sigma in _gaussians else torch.Tensor(np.exp(- ((tx - x0) ** 2 + (ty - y0) ** 2) / (2 * sigma ** 2)))
    _gaussians[sigma] = g

    # Determine the bounds of the source gaussian
    g_x_min, g_x_max = max(0, -x1), min(x2, w) - x1
    g_y_min, g_y_max = max(0, -y1), min(y2, h) - y1

    # Image range
    img_x_min, img_x_max = max(0, x1), min(x2, w)
    img_y_min, img_y_max = max(0, y1), min(y2, h)

    t[img_y_min:img_y_max, img_x_min:img_x_max] = g[g_y_min:g_y_max, g_x_min:g_x_max]

    return t

class MPIIDataset(Dataset):
    def __init__(self, transform=None):
        self.image_list = []
        self.heatmap_list = []
        self.transform = transform
        
        # load images
        self.root_dir = 'D:/data/mpii_data/'
        self.image_dir = self.root_dir + 'mpii_human_pose_v1/images/'
        annotation_dir = self.root_dir
        self.image_name_list = []
        self.joints_list = []
        

        with open(annotation_dir + 'mpii_annotations.json', 'r') as f:
            json_data = json.load(f)[:3000]

            for i in range(int(len(json_data))):
                if json_data[i]['img_width'] == 1280 and json_data[i]['img_height'] == 720 and json_data[i]['numOtherPeople'] == 0:
                    self.image_name_list.append(json_data[i]['img_paths'])
                    self.joints_list.append(np.array(json_data[i]['joint_self'])[:, :2])
            

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_dir + self.image_name_list[idx], 0)
        image = cv2.resize(image, dsize=(int(1280/2), int(704/2)))
        image = np.float32(image)
        image = image / 255
        
        temp_heatmap = []
        joint_points = self.joints_list[idx]
        joint_points[:, 0] = joint_points[:, 0] * 0.5
        joint_points[:, 1] = joint_points[:, 1] * ((704/2)/720)
        for i in range(len(joint_points)):
            heatmap = generate_gaussian(np.zeros((image.shape[0], image.shape[1])), joint_points[i][0], joint_points[i][1], sigma=5)
            temp_heatmap.append(heatmap)
        
        temp_heatmap = np.array(temp_heatmap, dtype=np.float32)
        gt = np.max(temp_heatmap, axis=0)
        
        # plt.subplot(1, 3, 1)
        # plt.imshow(image)
        # plt.subplot(1, 3, 2)
        # plt.imshow(gt)
        # plt.subplot(1, 3, 3)
        # plt.imshow(image)
        # peaks = peak_local_max(gt, min_distance=3, threshold_abs=0.5)
        # for peak in peaks:
        #     plt.plot(peak[1], peak[0], 'r.')
        # plt.show()
        
        return image, gt

# data = MPIIDataset()