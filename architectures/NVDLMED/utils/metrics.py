import numpy as np
from medpy.metric.binary import *


def calculate_3d_metrics(pred, gt):
    batch, c, _, _, _ = pred.size()
    pred_one_hot = (pred > 0.5).float()

    dsc_3d = np.zeros((batch, c))
    hd_3d = np.zeros((batch, c))
    for subj_i in range(batch):
        for tumor_class in range(c):
            single_class_pred = pred_one_hot[subj_i, tumor_class].cpu().numpy()
            single_class_gt = gt[subj_i, tumor_class].cpu().numpy()

            dsc_3d[subj_i, tumor_class] = dc(single_class_pred, single_class_gt)

            try:
                hd_3d[subj_i, tumor_class] = hd(single_class_pred, single_class_gt)
            except RuntimeError:
                hd_3d[subj_i, tumor_class] = float('nan')

    return dsc_3d, hd_3d
