from torch.utils.data import Dataset
from PIL import Image, ImageOps
from random import random
import os
import glob
import torch
import numpy as np


class ChaosDataset(Dataset):
    def __init__(self,
                 mode,
                 root_dir,
                 transform_input=None,
                 transform_mask=None,
                 augment=None,
                 equalize=False):

        self.root_dir = root_dir
        self.files = self.load_files(root_dir, mode)

        self.transform_input = transform_input
        self.transform_mask = transform_mask
        self.augment = augment
        self.equalize = equalize

    @staticmethod
    def load_files(root_dir, mode):
        assert mode in ["train", "val", "test"]
        files = []
        img_path = os.path.join(root_dir, mode, "Img")
        mask_path = os.path.join(root_dir, mode, "GT")

        images = os.listdir(img_path)
        images.sort()
        masks = os.listdir(mask_path)
        masks.sort()

        for img, mask in zip(images, masks):
            file = (os.path.join(img_path, img), os.path.join(mask_path, mask))
            files.append(file)

        return files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Load Pil image and mask from path
        img_path, mask_path = self.files[idx]
        file_name = os.path.abspath(mask_path).replace("\\", "/").split("/")[-1]

        img = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        # Apply transformations
        if self.augment:
            img, mask = self.augment(img, mask)

        if self.transform_input:
            img = self.transform_input(img)

        if self.transform_mask:
            mask = self.transform_mask(mask)
            mask = torch.from_numpy(mask).long()

        return img, mask, file_name


class GrayToClass(object):

    def __init__(self):
        self.class1 = 63    # Liver
        self.class2 = 126   # Kidney (R)
        self.class3 = 189   # Kidney (L)
        self.class4 = 252   # Spleen

    def __call__(self, mask):
        numpy_image = np.array(mask)

        numpy_image = np.where(numpy_image == self.class1, 1, numpy_image)
        numpy_image = np.where(numpy_image == self.class2, 2, numpy_image)
        numpy_image = np.where(numpy_image == self.class3, 3, numpy_image)
        numpy_image = np.where(numpy_image == self.class4, 4, numpy_image)

        return numpy_image


class Augment(object):

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img, mask):
        if random() > self.prob:
            img = ImageOps.flip(img)
            mask = ImageOps.flip(mask)

        if random() > self.prob:
            img = ImageOps.mirror(img)
            mask = ImageOps.mirror(mask)

        if random() > self.prob:
            angle = random() * 60 - 30
            img = img.rotate(angle)
            mask = mask.rotate(angle)

        return img, mask