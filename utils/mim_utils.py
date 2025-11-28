import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset
from PIL import Image
import os
import random


class MIMDataset(Dataset):
    """
    Custom Dataset that loads images from a text file list.
    It returns the raw image (we will mask it inside the training loop).
    """

    def __init__(self, txt_file, img_size=640):
        with open(txt_file, 'r') as f:
            self.img_paths = [line.strip() for line in f.readlines()]

        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            # Normalization helps convergence (ImageNet stats)
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        try:
            img = Image.open(path).convert('RGB')
            img = self.transform(img)
            return img
        except Exception as e:
            print(f"Error loading {path}: {e}")
            # Return a dummy image if file is corrupt
            return torch.zeros(3, 640, 640)


def generate_mask(batch_size, channels, height, width, patch_size=32, mask_ratio=0.6):
    """
    Generates a random binary mask for Grid Masking.
    1 = Keep, 0 = Drop.
    """
    assert height % patch_size == 0 and width % patch_size == 0

    h_patches = height // patch_size
    w_patches = width // patch_size
    num_patches = h_patches * w_patches
    num_masked = int(mask_ratio * num_patches)

    mask = torch.ones(batch_size, num_patches, device='cuda')

    # Randomly set indices to 0
    for i in range(batch_size):
        # Random permutation of indices, pick first 'num_masked' to be zero
        noise = torch.rand(num_patches, device='cuda')
        _, indices = torch.sort(noise)
        mask_indices = indices[:num_masked]
        mask[i, mask_indices] = 0

    # Reshape back to image grid
    mask = mask.view(batch_size, 1, h_patches, w_patches)
    # Upsample to pixel level
    mask = torch.nn.functional.interpolate(mask, scale_factor=patch_size, mode='nearest')

    return mask