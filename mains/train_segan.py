from architectures.SegAN.model.Segmentor import *
from architectures.SegAN.model.Critic import *
from datasets.MelanomaDataset import *
import torch
import torchvision.utils as vis
from torch.optim import Adam
import matplotlib.pyplot as plt

output_path = "./outputs"
if not os.path.exists(output_path):
    os.makedirs(output_path)

plot_path = "./plots/"
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

saved_models_path = "./saved_models/"
if not os.path.exists(saved_models_path):
    os.makedirs(saved_models_path)

data_path = "./datasets/data"
num_epoch = 300
lr = 0.003
lr_decay = 0.5
batch_size = 12
k_decay = 0.9
k = 1


def dice_score_and_jaccard(predicted, target):
    predicted_array = predicted.cpu().numpy()
    target_array = target.cpu().numpy()
    Jaccard, Dice = [], []
    for x in range(predicted.size()[0]):
        jaccard = np.sum(predicted_array[x][target_array[x] == 1]) / (
            np.sum(predicted_array[x]) + np.sum(target_array[x]) - np.sum(predicted_array[x][target_array[x] == 1]))
        dice = np.sum(predicted_array[x][target_array[x] == 1]) * 2 / (
            np.sum(predicted_array[x]) + np.sum(target_array[x]))
        Jaccard.append(jaccard)
        Dice.append(dice)

    return np.mean(Dice, axis=0), np.mean(Jaccard, axis=0)


def multi_scale_loss(input_clone, target, output, Critic):
    output_masked = mask_image(input_clone, output)
    predicted_C = Critic(output_masked)

    target_masked = mask_image(input_clone, target)
    target_C = Critic(target_masked)

    return torch.mean(torch.abs(predicted_C - target_C))


def dice_loss(predicted, target):
    num = predicted * target
    num = torch.sum(num, dim=2)
    num = torch.sum(num, dim=2)

    den1 = predicted * predicted
    den1 = torch.sum(den1, dim=2)
    den1 = torch.sum(den1, dim=2)

    den2 = target * target
    den2 = torch.sum(den2, dim=2)
    den2 = torch.sum(den2, dim=2)

    dice = 2 * (num / (den1 + den2))

    dice_total = torch.sum(dice) / dice.size(0)

    return dice_total


def update_optimizer(lr, Segmentor_params, Critic_params):
    optimizer_seg = Adam(Segmentor_params, lr=lr, betas=(0.5, 0.999))
    optimizer_crit = Adam(Critic_params, lr=lr, betas=(0.5, 0.999))
    return optimizer_seg, optimizer_crit


def save_checkpoints(input, label, predictions, output_path, epoch, is_train):
    id = ""
    if not is_train:
        id = "_val"
    vis.save_image(input.double(),
                      '%s/input%s_%d.png' % (output_path, id, epoch),
                      normalize=True)
    vis.save_image(label.double(),
                      '%s/label%s_%d.png' % (output_path, id, epoch),
                      normalize=True)
    vis.save_image(predictions.double(),
                      '%s/result%s_%d.png' % (output_path, id, epoch),
                      normalize=True)


def mask_image(input, mask):
    masked_image = input.clone()
    for d in range(input.shape[1]):
        masked_image[:, d, :, :] = (input[:, d, :, :].unsqueeze(1) * mask).squeeze()
    return masked_image.cuda()


def eval(model, data_loader, output_path, epoch=0):
    model.eval()
    with torch.no_grad():
        correct = 0
        dice_loss_eval = 0
        Dice, Jaccard = [], []
        for data, label in data_loader:
            data, label = data.cuda(), label.cuda()
            predictions = model(data)

            predictions = torch.sigmoid(predictions * k)
            predictions[predictions < 0.4] = 0
            predictions[predictions >= 0.4] = 1
            dice_loss_eval += dice_loss(predictions, label)
            correct += (predictions == label).sum().item() / label.nelement()
            dice_mean, jaccard_mean = dice_score_and_jaccard(predictions, label)
            Dice.append(dice_mean)
            Jaccard.append(jaccard_mean)

        if epoch % 20 == 0:
            save_checkpoints(data, label, predictions, output_path, epoch, is_train=False)

        dice_loss_eval = dice_loss_eval / len(data_loader)
        dice_score = np.array(Dice).mean()
        jaccard_index = np.array(Jaccard).mean()
        print("Dice_loss : {}".format(dice_loss_eval))
        print("Dice_score : {}".format(dice_score))
        print("Jaccard_index : {}".format(jaccard_index))
    return correct / len(data_loader), dice_score, jaccard_index


if __name__ == "__main__":

    ## build models
    Segmentor = Segmentor().cuda()
    Critic = Critic().cuda()

    # set optimizers
    optimizer_seg, optimizer_crit = update_optimizer(lr, Segmentor.parameters(), Critic.parameters())

    # Init training
    Segmentor.train()
    Critic.train()
    train_loader = loader(Dataset_train(data_path), batch_size)

    losses_dice_val = []
    losses_S_train = []
    losses_C_train = []
    train_accuracies = []
    val_accuracies = []
    val_dices = []
    val_jaccards = []
    max_jaccard = 0
    max_dice = 0

    for epoch in range(num_epoch):
        correct_train = 0
        loss_S_train = 0
        loss_C_train = 0
        for batch_idx, sample in enumerate(train_loader):

            # Train Critic
            Critic.zero_grad()
            input, target = sample[0].cuda(), sample[1].cuda()

            output = Segmentor(input)
            output = torch.sigmoid(output * k)
            output = output.detach()
            input_clone = input.clone()

            loss_C = - multi_scale_loss(input_clone, target, output, Critic)
            loss_C_train += loss_C.item()

            print("Loss C: {} ".format(loss_C.item()))
            loss_C.backward()
            optimizer_crit.step()

            # clip parameters in D
            for p in Critic.parameters():
                p.data.clamp_(-0.05, 0.05)

            # train Segmentor
            Segmentor.zero_grad()
            output = Segmentor(input)
            output = torch.sigmoid(output * k)

            loss_S = multi_scale_loss(input_clone, target, output, Critic)
            loss_S_train += loss_S.item()
            loss_dice_train = 1 - 1 * dice_loss(output, target)

            print("Loss dice: " + str(loss_dice_train.item()))
            loss_S_dice = loss_dice_train + loss_S
            print("Loss S: {} ".format(loss_S.item()))
            loss_S_dice.backward()
            optimizer_seg.step()

            # Accuracy
            output[output < 0.4] = 0
            output[output >= 0.4] = 1
            correct_train += (output == target).sum().item() / target.nelement()

        loss_C_train = loss_C_train / len(train_loader)
        losses_C_train.append(loss_C_train)

        print("--------- {} ---------".format(epoch))
        print("Train_S_loss : {}".format(loss_C_train))

        loss_S_train = loss_S_train / len(train_loader)
        losses_S_train.append(loss_S_train)
        print("Train_S_loss : {}".format(loss_S_train))

        train_accuracy = correct_train / len(train_loader)
        train_accuracies.append(train_accuracy)
        print("Train_accuracy : {}".format(train_accuracy))

        if epoch % 20 == 0:
            save_checkpoints(input, target, output, output_path, epoch, is_train=True)

        ### ------ Evaluation ------ ###

        val_loader = loader(Dataset_val_test(data_path, False), batch_size)
        val_accuracy, val_dice, val_jaccard = eval(Segmentor, val_loader, output_path, epoch)
        if max_dice < val_dice:
            torch.save(Segmentor, saved_models_path + 'Segmentor_max_dice.pt')
            max_dice = val_dice
        if max_jaccard < val_jaccard:
            torch.save(Segmentor, saved_models_path + 'Segmentor_max_jaccard.pt')
            max_jaccard = val_jaccard
        print("Val_accuracy : {}".format(val_accuracy))
        val_accuracies.append(val_accuracy)
        val_dices.append(val_dice)
        val_jaccards.append(val_jaccard)
        Segmentor.train()
        Critic.train()

        # Decrease learning rate and update
        if epoch % 25 == 0:
            lr = lr * lr_decay
            if k > 0.3:
                k = k * k_decay
            if lr <= 0.00000001:
                lr = 0.00000001
            optimizer_seg, optimizer_crit = update_optimizer(lr, Segmentor.parameters(), Critic.parameters())

    ### ------ Test ------ ###
    torch.save(Segmentor, saved_models_path + 'Segmentor_final.pt')
    test_loader = loader(Dataset_val_test(data_path, True), batch_size)

    test_accuracy, dice, jaccard = eval(Segmentor, test_loader, output_path)
    print("Test_accuracy : {}".format(test_accuracy))
    print("Max Validation Jaccard index : {}".format(max_jaccard))
    print("Max_Validation Dice Score : {}".format(max_dice))

    plt.title("Validation Dice Score")
    plt.xlabel("Epoch")
    plt.ylabel("Dice score")
    plt.plot(val_dices)
    plt.savefig(plot_path + 'dice.png')
    plt.show()

    plt.title("Validation Jaccard Index")
    plt.xlabel("Epoch")
    plt.ylabel("Jaccard Index")
    plt.plot(val_jaccards)
    plt.savefig(plot_path + 'jaccard.png')
    plt.show()

    plt.title("Training and Validation Losses for Critic and Segmentor")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(losses_C_train, label="Critic Loss")
    plt.plot(losses_S_train, label="Segmentor Loss")
    plt.legend(loc="lower right")
    plt.savefig(plot_path + 'losses.png')
    plt.show()

    plt.title("Training and Validation Accuracies for Segmentor")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.plot(train_accuracies, label="Train")
    plt.plot(val_accuracies, label="Validation")
    plt.legend(loc="lower right")
    plt.savefig(plot_path + 'accuracy.png')
    plt.show()
