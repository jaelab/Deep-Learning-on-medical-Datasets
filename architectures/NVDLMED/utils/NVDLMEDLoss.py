from torch import nn
import torch


class NVDLMEDLoss(nn.Module):
    def __init__(self):
        super(NVDLMEDLoss, self).__init__()
        self.loss_l2 = nn.MSELoss()

    def forward(self, output_gt, gt, output_vae, input, mu, var):
        l_dice = loss_dice(output_gt, gt)
        l_l2 = self.loss_l2(output_vae, input)
        l_kl = loss_kl(mu, var)

        return l_dice + 0.05 * l_l2 + 0.1 * l_kl, l_dice, l_l2, l_kl


def loss_dice(pred, gt, epsilon=1e-6):
    sum_dim = (2, 3, 4)
    intersect = (pred * gt).sum(sum_dim)
    denominator = (pred * pred).sum(sum_dim) + (gt * gt).sum(sum_dim)

    per_channel_dice = 2 * (intersect / denominator.clamp(min=epsilon))

    return (1. - per_channel_dice).sum(1).mean()


def loss_kl(z_mean, z_var):

    return torch.sum(z_var.exp() + z_mean.pow(2) - 1. - z_var)