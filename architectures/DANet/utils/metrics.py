from architectures.DANet.utils.utils import prediction_to_segmentation
import torch
import os
from architectures.DANet.utils.constant import *
import numpy as np
import nibabel as nib
from medpy.metric.binary import *


def get_onehot_segmentation(target):
    batch_size, height, width = target.size()
    one_hot = torch.zeros(batch_size, 5, height, width, dtype=torch.float).cuda()

    return one_hot.scatter_(1, target.unsqueeze(1), 1.0)


def dice_score(pred, target):
    pred_onehot = prediction_to_segmentation(pred)
    target_onehot = get_onehot_segmentation(target)

    dims = (0, 2, 3)
    intersection = torch.sum(pred_onehot * target_onehot, dims)
    cardinality = torch.sum(pred_onehot + target_onehot, dims)

    dice = (2. * intersection + 1e-8) / (cardinality + 1e-8)

    return dice


def volume_similarity(pred, target):
    a = np.count_nonzero(pred)
    b = np.count_nonzero(target)

    return 1.0 - abs(a - b) / (a + b)


def calculate_3d_metrics(volume_path):
    pred_volume_files = os.listdir(volume_path + "/Pred")
    gt_volume_files = os.listdir(volume_path + "/GT")

    dsc_3d = np.zeros((len(pred_volume_files), NUMBER_CLASS))
    assd_3d = np.zeros((len(pred_volume_files), NUMBER_CLASS))
    vs_3d = np.zeros((len(pred_volume_files), NUMBER_CLASS))
    for subj_i in range(len(pred_volume_files)):
        pred_path = os.path.join(volume_path + "/Pred", pred_volume_files[subj_i])
        gt_path = os.path.join(volume_path + "/GT", gt_volume_files[subj_i])

        pred_volume = nib.load(pred_path).get_data()
        gt_volume = nib.load(gt_path).get_data()

        for organ_class in range(NUMBER_CLASS):
            single_organ_pred = np.zeros(pred_volume.shape, dtype=np.int8)
            single_organ_gt = np.zeros(gt_volume.shape, dtype=np.int8)

            idx_pred = np.where(pred_volume == organ_class + 1)
            single_organ_pred[idx_pred] = 1

            idx_gt = np.where(gt_volume == organ_class + 1)
            single_organ_gt[idx_gt] = 1

            dsc_3d[subj_i, organ_class] = dc(single_organ_pred, single_organ_gt)
            assd_3d[subj_i, organ_class] = assd(single_organ_pred, single_organ_gt) if np.count_nonzero(single_organ_pred) != 0 else float('nan')
            vs_3d[subj_i, organ_class] = volume_similarity(single_organ_pred, single_organ_gt)

    return dsc_3d, assd_3d, vs_3d
