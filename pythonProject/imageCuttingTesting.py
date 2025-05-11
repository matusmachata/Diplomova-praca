import os
import random
import torch
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import tifffile as tiff
from glob import glob
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np
from torch.optim import lr_scheduler
import re
from pathlib import Path
import ssl
from torch.optim.lr_scheduler import ReduceLROnPlateau
import cv2



# image_path = 'C:/Users/Matus/Desktop/FEI/dataset_OIA_DDR/DDR-dataset/lesion_segmentation/test/image/007-3454-200.jpg'  # Replace with your image path
# image_path = 'C:/Users/Matus/Desktop/FEI/dataset_OIA_DDR/DDR-dataset/lesion_segmentation/test/image/007-3489-200.jpg'
# image_path = 'C:/Users/Matus/Desktop/FEI/dataset_OIA_DDR/DDR-dataset/lesion_segmentation/test/image/007-3494-200.jpg'
# image_path = 'C:/Users/Matus/Desktop/FEI/dataset_OIA_DDR/DDR-dataset/lesion_segmentation/test/image/007-3722-200.jpg'
# image_path = 'C:/Users/Matus/Desktop/FEI/dataset_OIA_DDR/DDR-dataset/lesion_segmentation/test/image/20170627170651362.jpg'
image_path = 'C:/Users/Matus/Desktop/FEI/dataset_OIA_DDR/DDR-dataset/lesion_segmentation/test/image/20170505161143089.jpg'

# image_mask_path = 'C:/Users/Matus/Desktop/FEI/dataset_OIA_DDR/DDR-dataset/lesion_segmentation/test/label/EX/007-3489-200.tif'
# image_mask_path = 'C:/Users/Matus/Desktop/FEI/dataset_OIA_DDR/DDR-dataset/lesion_segmentation/test/label/EX/007-3494-200.tif'
# image_mask_path = 'C:/Users/Matus/Desktop/FEI/dataset_OIA_DDR/DDR-dataset/lesion_segmentation/test/label/EX/007-3722-200.tif'
image_mask_path = 'C:/Users/Matus/Desktop/FEI/dataset_OIA_DDR/DDR-dataset/lesion_segmentation/test/label/SE/20170627170651362.tif'


def crop_and_resize(image, mask, size):
    """Crop image and mask to a square (max possible size) and resize to target size."""
    # Get original dimensions
    w, h = image.size
    min_dim = min(w, h)

    # Calculate crop box (center crop to square)
    left = (w - min_dim) // 2
    top = (h - min_dim) // 2
    right = left + min_dim
    bottom = top + min_dim

    # Crop and resize
    image = image.crop((left, top, right, bottom)).resize(size, Image.BILINEAR)
    mask = Image.fromarray(mask).crop((left, top, right, bottom)).resize(size, Image.NEAREST)

    return image, mask


def cut_image_pattern(image, patch_size, height, width):
    """Cuts the image into patches based on the red-line pattern."""
    step_size = patch_size // 2
    patches = []

    # Convert PIL Image to NumPy array
    image = np.array(image)

    for y in range(0, height, step_size):
        for x in range(0, width, step_size):
            if ((x // step_size) + (y // step_size)) % 2 == 0:  # Red pattern condition
                if x + patch_size <= width and y + patch_size <= height:
                    patch = image[y:y + patch_size, x:x + patch_size]
                    patches.append(patch)

    return patches

image_size = (32*18, 32*18)
patch_size=64

image = Image.open(image_path).convert("RGB")
mask = tiff.imread(image_mask_path)

image, mask = crop_and_resize(image, mask, image_size)

# Cut into patches
image_patches = cut_image_pattern(image, patch_size, image_size[0], image_size[1])
mask_patches = cut_image_pattern(mask, patch_size, image_size[0],image_size[1])  # Ensure single channel mask

for i,im in enumerate(image_patches):
    plt.figure()
    plt.imshow(im)
    plt.show()