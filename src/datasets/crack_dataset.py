import os
from glob import glob

import cv2
import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset


class CrackDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform

        image_paths = glob(os.path.join(images_dir, "*.jpg"))
        mask_paths = glob(os.path.join(masks_dir, "*.mat"))

        image_map = {
            os.path.splitext(os.path.basename(path))[0]: path
            for path in image_paths
        }
        mask_map = {
            os.path.splitext(os.path.basename(path))[0]: path
            for path in mask_paths
        }

        common_ids = sorted(set(image_map.keys()) & set(mask_map.keys()), key=int)

        if len(common_ids) == 0:
            raise ValueError("No matching image-mask pairs found.")

        self.image_paths = [image_map[i] for i in common_ids]
        self.mask_paths = [mask_map[i] for i in common_ids]

        print(f"Matched image-mask pairs: {len(self.image_paths)}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mat = loadmat(mask_path)
        mask = mat["groundTruth"][0, 0][1]

        mask = mask.astype(np.float32)

        if mask.max() > 1:
            mask = mask / 255.0

        mask = np.expand_dims(mask, axis=-1)

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        if isinstance(mask, np.ndarray):
            mask = torch.tensor(mask, dtype=torch.float32)

        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        elif mask.ndim == 3:
            mask = mask.permute(2, 0, 1)

        return image, mask