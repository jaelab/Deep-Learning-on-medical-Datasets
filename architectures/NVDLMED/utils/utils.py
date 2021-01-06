import nibabel as nib
import numpy as np
import os
import torch


def prediction_to_nii(pred, gt, input, img_names, out_path):
    pred_one_hot = (pred > 0.5).float()
    pred_seg = channels_to_segmentation(pred_one_hot)
    gt_seg = channels_to_segmentation(gt)

    for i, name in enumerate(img_names):
        patient_pred = pred_seg[i].cpu().numpy()
        patient_gt = gt_seg[i].cpu().numpy()
        patient_input = input[i].cpu().numpy()

        nii_pred = nib.Nifti1Image(patient_pred, affine=np.eye(4))
        nii_gt = nib.Nifti1Image(patient_gt, affine=np.eye(4))
        nii_input = nib.Nifti1Image(patient_input, affine=np.eye(4))

        nib.save(nii_pred, os.path.join(out_path, name + "_pred"))
        nib.save(nii_gt, os.path.join(out_path, name + "_gt"))
        nib.save(nii_input, os.path.join(out_path, name + "_input"))


def channels_to_segmentation(one_hot_tensor):
    brats_pixel_values = np.array([1, 2, 3])
    brats_pixel = torch.from_numpy(brats_pixel_values).cuda()

    out = one_hot_tensor * brats_pixel.view(1, 3, 1, 1, 1)

    return out.max(dim=1)[0]
