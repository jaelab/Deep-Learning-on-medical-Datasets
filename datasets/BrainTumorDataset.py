import SimpleITK as sitk
import torch
import re
import os
from torch.utils.data import Dataset
from scipy.ndimage import zoom
import glob
import numpy as np
from architectures.NVDLMED.utils.constant import *


class BrainTumorDataset(Dataset):

    def __init__(self,
                 mode="train",
                 data_path="../rawdata/brats",
                 desired_resolution=(80, 96, 64),
                 original_resolution=(155, 240, 240),
                 transform_input=None,
                 transform_gt=None):
        self.mode = mode
        self.data_path = os.path.join(data_path, mode)
        self.data_files = {"t1": glob.glob(self.data_path + '/*/*t1.nii.gz'),
                               "t2": glob.glob(self.data_path + '/*/*t2.nii.gz'),
                               "flair": glob.glob(self.data_path + '/*/*flair.nii.gz'),
                               "t1ce": glob.glob(self.data_path + '/*/*t1ce.nii.gz'),
                               "seg": glob.glob(self.data_path + '/*/*seg.nii.gz')}

        self.desired_resolution = desired_resolution
        self.original_resolution = original_resolution
        self.transform_input = transform_input
        self.transform_gt = transform_gt
        self.files = self.find_files()

    def find_files(self):
        path = re.compile('.*_(\w*)\.nii\.gz')
        data_paths = [{
            path.findall(item)[0]: item
            for item in items
        }
            for items in list(zip(self.data_files["t1"], self.data_files["t2"], self.data_files["t1ce"], self.data_files["flair"], self.data_files["seg"]))]

        return data_paths

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data_files = self.files[idx]
        numpy_data = np.array([sitk.GetArrayFromImage(sitk.ReadImage(file))
                               for file in data_files.values()], dtype=np.float32)
        input = self.transform_input(numpy_data[0:4])
        gt = self.transform_gt(numpy_data[-1])

        if self.mode == "test":
            seg_name = data_files["seg"].replace("\\", "/").split("/")[-2]
            return torch.from_numpy(input), torch.from_numpy(gt), seg_name
        else:
            return torch.from_numpy(input), torch.from_numpy(gt)


class Resize(object):
    def __init__(self, factors, mode='constant', dtype=np.float32):
        self.factors = factors
        self.mode = mode
        self.dtype = dtype

    def __call__(self, data):
        assert len(self.factors) == 3
        resized_data = np.array([zoom(data[i], self.factors, mode=self.mode)
                                 for i in range(data.shape[0])], dtype=self.dtype)

        return resized_data


class Normalize(object):
    def __init__(self):
        pass

    def __call__(self, data):
        normalized_data = np.array([(data[i] - data[i].mean()) / data[i].std()
                                    for i in range(data.shape[0])], dtype=np.float32)

        return normalized_data


class Labelize(object):
    def __init__(self):
        pass

    def __call__(self, data):
        wt = (data == MASK_NET) + (data == MASK_ED) + (data == MASK_ET)  # Whole tumor (NET + ED + ET)
        tc = (data == MASK_NET) + (data == MASK_ET)   # Tumor core (NET + ET)
        et = data == MASK_ET  # Enhancing tumor (ET)

        return np.array([wt, tc, et], dtype=np.uint8)