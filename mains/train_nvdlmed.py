import sys

sys.path.append("../INF8225-Project/")

from architectures.NVDLMED.model.NVDLMED import *
from datasets.BrainTumorDataset import *
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim import Adam
from torch.optim.lr_scheduler import *
from architectures.NVDLMED.utils.NVDLMEDLoss import *
from tqdm import tqdm
from architectures.NVDLMED.utils.metrics import *
import warnings
import argparse
import shutil
from architectures.NVDLMED.utils.utils import *
from architectures.NVDLMED.utils.HierarchyCreator import create_hierarchy
from architectures.NVDLMED.utils.constant import *

warnings.filterwarnings("ignore")


def run_training(args):
    desired_resolution = (args.desired_resolution_h, args.desired_resolution_w, args.desired_resolution_d)
    factor = (desired_resolution[0] / ORIGINAL_RES_H, desired_resolution[1] / ORIGINAL_RES_W, desired_resolution[2] / ORIGINAL_RES_D)

    train_dataset = BrainTumorDataset(
        mode="train",
        data_path=args.root_dir,
        desired_resolution=desired_resolution,
        original_resolution=(ORIGINAL_RES_H, ORIGINAL_RES_W, ORIGINAL_RES_D),
        transform_input=transforms.Compose([Resize(factor), Normalize()]),
        transform_gt=transforms.Compose([Labelize(), Resize(factor, mode="nearest")]))

    val_dataset = BrainTumorDataset(
        mode="val",
        data_path=args.root_dir,
        desired_resolution=desired_resolution,
        original_resolution=(ORIGINAL_RES_H, ORIGINAL_RES_W, ORIGINAL_RES_D),
        transform_input=transforms.Compose([Resize(factor), Normalize()]),
        transform_gt=transforms.Compose([Labelize(), Resize(factor, mode="nearest")]))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    net = NVDLMED(input_shape=(4,) + desired_resolution)
    net.cuda()

    NVDLMED_loss = NVDLMEDLoss()
    optimizer = Adam(net.parameters(), lr=args.lr, weight_decay=1e-5)

    lambda1 = lambda epoch: (1 - epoch / args.epochs) ** 0.9
    scheduler = LambdaLR(optimizer, lr_lambda=lambda1)

    loss_train, loss_dice, loss_l2, loss_kl = [], [], [], []
    dsc_train, dsc_val = [], []
    hd_train, hd_val = [], []

    best_dice_3d, best_epoch = 0, 0
    for epoch in range(args.epochs):

        # Training loop
        net.train()
        loss_train_batch, loss_dice_batch, loss_l2_batch, loss_kl_batch = [], [], [], []
        dsc_train_batch, hd_train_batch = [], []
        with tqdm(total=len(train_loader), ascii=True) as training_bar:
            training_bar.set_description(f'[Training] Epoch {epoch + 1}')

            for (input, target) in train_loader:
                input, target = input.cuda(), target.cuda()

                optimizer.zero_grad()
                output_gt, output_vae, mu, logvar = net(input)
                loss, l_dice, l_l2, l_kl = NVDLMED_loss(output_gt, target, output_vae, input, mu, logvar)

                loss.backward()
                optimizer.step()

                loss_train_batch.append(loss.item())
                loss_dice_batch.append(l_dice.item())
                loss_l2_batch.append(l_l2.item())
                loss_kl_batch.append(l_kl.item())

                dsc_3d, hd_3d = calculate_3d_metrics(output_gt, target)
                dsc_train_batch.extend(dsc_3d)
                hd_train_batch.extend(hd_3d)

                training_bar.set_postfix_str(
                    "Loss: {:.3f} | Dice loss: {:.3f}, L2 loss: {:.3f}, KL loss {:.3f}"
                        .format(loss.item(), l_dice.item(), l_l2.item(), l_kl.item()))
                training_bar.update()

            training_bar.set_postfix_str("Mean loss: {:.4f}".format(np.mean(loss_train_batch)))

        # Validation loop
        net.eval()
        dsc_val_batch, hd_val_batch = [], []
        with tqdm(total=len(valid_loader), ascii=True) as val_bar:
            val_bar.set_description('[Validation]')

            for (input, target) in valid_loader:
                input, target = input.cuda(), target.cuda()

                with torch.no_grad():
                    output_gt = net(input)

                    dsc_3d, hd_3d = calculate_3d_metrics(output_gt, target)
                    dsc_val_batch.extend(dsc_3d)
                    hd_val_batch.extend(hd_3d)

                val_bar.update()

            val_bar.set_postfix_str(
                "Dice 3D: {:.3f} || WT: {:.3f}, TC: {:.3f}, ET: {:.3f}"
                    .format(np.mean(dsc_val_batch), np.mean(dsc_val_batch, 0)[0], np.mean(dsc_val_batch, 0)[1], np.mean(dsc_val_batch, 0)[2])
            )

        current_dice_3d = np.mean(dsc_val_batch)
        if current_dice_3d > best_dice_3d:
            best_dice_3d = current_dice_3d
            torch.save(net.state_dict(), args.root_dir + "/save/net.pth")

        scheduler.step()

        # Save Statistics
        loss_train.append(loss_train_batch)
        loss_dice.append(loss_dice_batch)
        loss_l2.append(loss_l2_batch)
        loss_kl.append(loss_kl_batch)
        dsc_train.append(dsc_train_batch)
        hd_train.append(hd_train_batch)
        dsc_val.append(dsc_val_batch)
        hd_val.append(hd_val_batch)

        np.save(args.root_dir + "/save/loss", loss_train)
        np.save(args.root_dir + "/save/loss_dice", loss_dice)
        np.save(args.root_dir + "/save/loss_l2", loss_l2)
        np.save(args.root_dir + "/save/loss_kl", loss_kl)
        np.save(args.root_dir + "/save/dsc_train", dsc_train)
        np.save(args.root_dir + "/save/hd_train", hd_train)
        np.save(args.root_dir + "/save/dsc_val", dsc_val)
        np.save(args.root_dir + "/save/hd_val", hd_val)


def run_eval(args):
    desired_resolution = (args.desired_resolution_h, args.desired_resolution_w, args.desired_resolution_d)
    factor = (desired_resolution[0] / ORIGINAL_RES_H, desired_resolution[1] / ORIGINAL_RES_W, desired_resolution[2] / ORIGINAL_RES_D)

    test_dataset = BrainTumorDataset(
        mode="test",
        data_path=args.root_dir,
        desired_resolution=desired_resolution,
        original_resolution=(ORIGINAL_RES_H, ORIGINAL_RES_W, ORIGINAL_RES_D),
        transform_input=transforms.Compose([Resize(factor), Normalize()]),
        transform_gt=transforms.Compose([Labelize(), Resize(factor, mode="nearest")]))

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    net = NVDLMED(input_shape=(4,) + desired_resolution).cuda()
    net.load_state_dict(torch.load(args.checkpoint_path))

    net.eval()
    dsc_test, hd_test = [], []
    with tqdm(total=len(test_loader), ascii=True, position=0) as test_bar:
        test_bar.set_description('[Evaluation]')

        for (input, target, img_name) in test_loader:
            input, target = input.cuda(), target.cuda()

            with torch.no_grad():
                output_gt = net(input)
                prediction_to_nii(output_gt, target, input, img_name, args.root_dir + "/save/pred/")

                dsc_3d, hd_3d = calculate_3d_metrics(output_gt, target)
                dsc_test.extend(dsc_3d)
                hd_test.extend(hd_3d)

            test_bar.update()

        test_bar.set_postfix_str(
            "Dice: {:.3f} | HD: {:.3f}"
                .format(np.mean(dsc_test), np.mean(hd_test))
        )

        np.save(args.root_dir + "/save/dsc_test", dsc_test)
        np.save(args.root_dir + "/save/hd_test", hd_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', default='../rawdata/MICCAI_BraTS_2018_Data_Training', type=str)
    parser.add_argument('--root_dir', default='../rawdata/brats', type=str)
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--desired_resolution_h', default=80, type=int)
    parser.add_argument('--desired_resolution_w', default=96, type=int)
    parser.add_argument('--desired_resolution_d', default=64, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--epochs', default=150, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--train', default=False, action='store_true')
    parser.add_argument('--eval', default=False, action='store_true')
    parser.add_argument('--create_hierarchy', default=False, action='store_true')
    parser.add_argument('--checkpoint_path', default='../rawdata/brats/save/net.pth', type=str)

    args = parser.parse_args()

    assert args.desired_resolution_h % 16 == 0
    assert args.desired_resolution_w % 16 == 0
    assert args.desired_resolution_d % 16 == 0

    if args.create_hierarchy:
        print("Creating folders for the model!\n")
        shutil.rmtree(args.root_dir, ignore_errors=True)
        create_hierarchy(data_dir=args.data_dir, out_dir=args.root_dir)

    if args.train:
        run_training(args)

    if args.eval:
        run_eval(args)
