from torch import nn


class MSDualGuidedLoss(nn.Module):
    def __init__(self):
        super(MSDualGuidedLoss, self).__init__()

        self.softmax = nn.Softmax()
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

    def forward(self, semVector1, semVector2, fsms, fai, semModule1, semModule2, predict1, predict2, mask):
        predict_loss = sum([self.ce_loss(predict1[i], mask) + self.ce_loss(predict2[i], mask) for i in range(len(predict1))])
        sementic_loss = sum([self.mse_loss(semVector1[i], semVector2[i]) for i in range(len(semVector1))])
        reconst_loss = sum([self.mse_loss(fsms[i], semModule1[i]) + self.mse_loss(fai[i], semModule2[i]) for i in range(len(semModule1))])

        total_loss = predict_loss + 0.25 * sementic_loss + 0.1 * reconst_loss

        return total_loss
