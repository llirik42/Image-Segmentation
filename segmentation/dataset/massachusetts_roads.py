import os
import random

import cv2
import torch
from torch.utils.data import Dataset
from torch import tensor
import numpy as np
from PIL import Image

from .config import config

from torchvision.transforms import Normalize
import torchvision.transforms.functional as TF
from torchvision import transforms


def get_all_file_paths(directory):
    file_paths = []
    for root, directories, files in os.walk(directory):
        for filename in files:
            filepath = str(os.path.join(root, filename))
            file_paths.append(filepath)

    # path/124.tiff, path/123.tiff, ... - path/123.tiff, path/124.tiff, ...
    return sorted(file_paths, key=lambda x: int(x.split('/')[-1].split('.')[0]))


class MassachusettsRoadsDataset(Dataset):
    def __init__(self, train=True, transforms=None):
        if train:
            self.image_paths = get_all_file_paths(config['save_path'] + "/train_sample")
            self.mask_paths = get_all_file_paths(config['save_path'] + "/train_target")
        else:
            self.image_paths = get_all_file_paths(config['save_path'] + "/test_sample")
            self.mask_paths = get_all_file_paths(config['save_path'] + "/test_target")

        assert len(self.image_paths) == len(self.mask_paths)

        self.transforms = transforms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0

        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE).astype(np.float32)/255.0

        image = tensor(image, dtype=torch.float32).permute(2, 1, 0)
        mask = tensor(mask, dtype=torch.float32).unsqueeze(0)

        if self.transforms:
            image = self.transforms(image)
            mask = self.transforms(mask)

        mask = torch.where(mask > 0.5, torch.tensor(1.0), torch.tensor(0.0))

        return image, mask
