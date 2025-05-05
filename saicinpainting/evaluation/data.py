import glob
import os
import logging

import cv2
import PIL.Image as Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from saicinpainting.training.data.masks import get_mask_generator

LOGGER = logging.getLogger(__name__)

def load_image(fname, mode='RGB', return_orig=False):
    img = cv2.imread(fname)
    if img is None:
        raise ValueError(f'Cannot read image {fname}')
    if mode == 'RGB':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if return_orig:
        return img.astype('float32') / 255, img
    return img.astype('float32') / 255

def ceil_modulo(x, mod):
    if x % mod == 0:
        return x
    return (x // mod + 1) * mod

def pad_img_to_modulo(img, mod):
    channels, height, width = img.shape
    out_height = ceil_modulo(height, mod)
    out_width = ceil_modulo(width, mod)
    return np.pad(img, ((0, 0), (0, out_height - height), (0, out_width - width)), mode='reflect')

def pad_tensor_to_modulo(img, mod):
    batch_size, channels, height, width = img.shape
    out_height = ceil_modulo(height, mod)
    out_width = ceil_modulo(width, mod)
    return F.pad(img, pad=(0, out_width - width, 0, out_height - height), mode='reflect')

def scale_image(img, factor, interpolation=cv2.INTER_AREA):
    if img.shape[0] == 1:
        img = img[0]
    else:
        img = np.transpose(img, (1, 2, 0))

    img = cv2.resize(img, dsize=None, fx=factor, fy=factor, interpolation=interpolation)

    if img.ndim == 2:
        img = img[None, ...]
    else:
        img = np.transpose(img, (2, 0, 1))
    return img

class InpaintingDataset(Dataset):
    def __init__(self, indir, mask_generator=None, transform=None, img_suffix='.jpg', pad_out_to_modulo=None):
        self.in_files = list(glob.glob(os.path.join(indir, '**', f'*{img_suffix}'), recursive=True))
        self.mask_generator = mask_generator
        self.transform = transform
        self.pad_out_to_modulo = pad_out_to_modulo

    def __len__(self):
        return len(self.in_files)

    def __getitem__(self, i):
        path = self.in_files[i]
        img, raw_image = load_image(path, return_orig=True)

        if self.transform is not None:
            img = self.transform(image=img)['image']

        img = np.transpose(img, (2, 0, 1))
        
        if self.mask_generator is not None:
            mask = self.mask_generator(img, raw_image=raw_image, image_path=path)
        else:
            mask = np.zeros((1, *img.shape[1:]), dtype=np.float32)

        if self.pad_out_to_modulo is not None and self.pad_out_to_modulo > 1:
            img = pad_img_to_modulo(img, self.pad_out_to_modulo)
            mask = pad_img_to_modulo(mask, self.pad_out_to_modulo)

        return dict(image=img, mask=mask, image_path=path)

class OurInpaintingDataset(Dataset):
    def __init__(self, indir, mask_generator=None, transform=None, img_suffix='.png', pad_out_to_modulo=None):
        # Get all image files with the specified suffix
        all_files = list(glob.glob(os.path.join(indir, '**', f'*{img_suffix}'), recursive=True))
        
        # Filter out files that are likely masks based on common naming patterns
        # This prevents the dataset from trying to process mask images as input images
        self.in_files = []
        for file_path in all_files:
            filename = os.path.basename(file_path)
            # Skip files with common mask naming patterns
            if ('_mask' in filename.lower() or 
                'mask_' in filename.lower() or 
                'mask' in filename.lower()):
                continue
            self.in_files.append(file_path)
        
        self.mask_generator = mask_generator
        self.transform = transform
        self.pad_out_to_modulo = pad_out_to_modulo
        
        # Log info about filtered files
        if len(all_files) - len(self.in_files) > 0:
            LOGGER.info(f"Filtered out {len(all_files) - len(self.in_files)} mask images from dataset")

    def __len__(self):
        return len(self.in_files)

    def __getitem__(self, i):
        path = self.in_files[i]
        img, raw_image = load_image(path, return_orig=True)
        
        if self.transform is not None:
            img = self.transform(image=img)['image']
            
        img = np.transpose(img, (2, 0, 1))
        
        if self.mask_generator is not None:
            mask = self.mask_generator(img, raw_image=raw_image, image_path=path)
        else:
            mask = np.zeros((1, img.shape[1], img.shape[2]), dtype=np.float32)
        
        # Store original image dimensions before padding
        orig_height, orig_width = img.shape[1], img.shape[2]
            
        if self.pad_out_to_modulo is not None and self.pad_out_to_modulo > 1:
            img = pad_img_to_modulo(img, self.pad_out_to_modulo)
            mask = pad_img_to_modulo(mask, self.pad_out_to_modulo)
            
        # Return the original dimensions as unpad_to_size
        return dict(image=img, mask=mask, image_path=path, unpad_to_size=(orig_height, orig_width))

class InpaintingEvalOnlineDataset(Dataset):
    def __init__(self, indir, mask_generator, img_suffix='.jpg', pad_out_to_modulo=None, transform=None, out_size=None):
        self.in_files = list(glob.glob(os.path.join(indir, '**', f'*{img_suffix}'), recursive=True))
        self.mask_generator = mask_generator
        self.transform = transform
        self.pad_out_to_modulo = pad_out_to_modulo
        self.out_size = out_size

    def __len__(self):
        return len(self.in_files)

    def __getitem__(self, i):
        path = self.in_files[i]
        img, raw_image = load_image(path, return_orig=True)
        
        if self.transform is not None:
            img = self.transform(image=img)['image']
        
        if self.out_size is not None:
            img = cv2.resize(img, (self.out_size, self.out_size))
            
        img = np.transpose(img, (2, 0, 1))
        
        if self.mask_generator is not None:
            mask = self.mask_generator(img, raw_image=raw_image, image_path=path)
        else:
            mask = np.zeros((1, *img.shape[1:]), dtype=np.float32)
            
        if self.pad_out_to_modulo is not None and self.pad_out_to_modulo > 1:
            img = pad_img_to_modulo(img, self.pad_out_to_modulo)
            mask = pad_img_to_modulo(mask, self.pad_out_to_modulo)
            
        return dict(image=img, mask=mask, image_path=path)