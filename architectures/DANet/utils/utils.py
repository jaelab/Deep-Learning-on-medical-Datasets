import nibabel as nib
from PIL import Image
import torch
import os
import torch.nn.functional as F
import torchvision
from architectures.DANet.utils.constant import *
import numpy as np
import re


def prediction_to_segmentation(pred):
    soft_pred = F.softmax(pred)

    Max = soft_pred.max(dim=1, keepdim=True)[0]
    x = soft_pred / Max

    return (x == 1).float()


def prediction_to_normalized_pil(pred_onehot):
    chaos_pixel_values = np.array([MASK_BG, MASK_LIVER, MASK_KR, MASK_KL, MASK_SPLEEN])
    chaos_pixel = torch.from_numpy(chaos_pixel_values).cuda()

    out = pred_onehot * chaos_pixel.view(1, 5, 1, 1)

    return out.sum(dim=1, keepdim=True)


def prediction_to_png(pred, img_name, out_path):
    batch_size = pred.size(0)

    pred_onehot = prediction_to_segmentation(pred)
    normalized_pil = prediction_to_normalized_pil(pred_onehot)

    for i in range(batch_size):
        torchvision.utils.save_image(normalized_pil[i].data, os.path.join(out_path, img_name[i]))


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def find_images(path, regex=".*"):
    image_names = []
    if os.path.exists(path):
        for file in os.listdir(path):
            if file.endswith(".png") and re.match(regex, file):
                image_names.append(os.path.join(path, file))

    image_names.sort(key=natural_keys)

    return image_names


def create_nii_from_images(img_names, out_path):
    vol_numpy = np.zeros((IMG_HEIGTH, IMG_WIDTH, len(img_names)))
    for i, file in enumerate(img_names):
        imagePIL = np.array(Image.open(file).convert('L'))
        vol_numpy[:, :, i] = imagePIL / CLASS_INCREMENT

    xform = np.eye(4) * 2
    imgNifti = nib.nifti1.Nifti1Image(vol_numpy, xform)

    nib.save(imgNifti, out_path)


def create_3d_volume(path, out_path):
    files = os.listdir(path)
    subj_no = set([re.match("Subj_[0-9]*", file).group() for file in files])

    for subj in subj_no:
        img_names = find_images(path, subj)
        create_nii_from_images(img_names, out_path + "/" + subj)


