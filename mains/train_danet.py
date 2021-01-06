import sys

sys.path.append("../INF8225-Project/")

from architectures.DANet.model.MSDualGuided import *
from torch.utils.data import DataLoader
from datasets.ChaosDataset import *
from torchvision import transforms
from torch.optim import Adam
from architectures.DANet.utils.MSDualGuidedLoss import *
from tqdm import tqdm
from architectures.DANet.utils.metrics import *
import warnings
import shutil
import argparse
from architectures.DANet.utils.HierarchyCreator import *

warnings.filterwarnings("ignore")


def run_training(args):
    transform = transforms.Compose([transforms.ToTensor()])
    train_chaos_dataset = ChaosDataset(mode="train", root_dir=args.root_dir, transform_input=transform,
                                       transform_mask=GrayToClass(), augment=Augment())
    train_loader = DataLoader(train_chaos_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                              shuffle=True)

    val_chaos_dataset = ChaosDataset(mode="val", root_dir=args.root_dir, transform_input=transform,
                                     transform_mask=GrayToClass(), augment=None)
    val_loader = DataLoader(val_chaos_dataset, batch_size=16, num_workers=args.num_workers, shuffle=False)

    net = MSDualGuided().cuda()
    loss_module = MSDualGuidedLoss()

    lr = args.lr
    optimizer = Adam(net.parameters(), lr=lr, betas=(0.9, 0.99), amsgrad=False)

    lossG = []
    dsc = []
    assd = []
    vs = []

    best_dice_3d, best_epoch = 0, 0
    for i in range(args.epochs):

        # Training Loop
        with tqdm(total=len(train_loader), ascii=True) as training_bar:
            training_bar.set_description(f'[Training] Epoch {i + 1}')

            net.train()
            loss_train = 0
            for (image, mask, _) in train_loader:
                image, mask = image.cuda(), mask.cuda()
                semVector1, semVector2, fsms, fai, semModule1, semModule2, predict1, predict2 = net(image)

                optimizer.zero_grad()
                loss = loss_module(semVector1, semVector2, fsms, fai, semModule1, semModule2, predict1, predict2, mask)

                loss.backward()
                optimizer.step()

                loss_train += loss.item()

                segmentation_prediction = sum(list(predict1) + list(predict2)) / 8
                classes_dice = dice_score(segmentation_prediction, mask)

                training_bar.set_postfix_str(
                    "Mean dice: {:.3f} || Liver: {:.3f}, Kidney(R): {:.3f}, Kidney(L): {:.3f}, Spleen: {:.3f}"
                        .format(torch.mean(classes_dice[1:]), classes_dice[1], classes_dice[2], classes_dice[3],
                                classes_dice[4])
                )
                training_bar.update()

            training_bar.set_postfix_str("Mean loss: {:.4f}".format(loss_train / len(train_loader)))
            del semVector1, semVector2, fsms, fai, semModule1, semModule2, predict1, predict2

        # Validation Loop
        with tqdm(total=len(val_loader), ascii=True) as val_bar:
            val_bar.set_description('[Validation]')

            net.eval()
            for j, (val_image, val_mask, val_img_name) in enumerate(val_loader):
                val_image, val_mask = val_image.cuda(), val_mask.cuda()

                with torch.no_grad():
                    seg_pred = net(val_image)
                    prediction_to_png(seg_pred, val_img_name, out_path=args.root_dir + "/val/Result")

                val_bar.update()

            create_3d_volume(args.root_dir + "/val/Result", args.root_dir + "/val/Volume/Pred")
            dsc_3d, assd_3d, vs_3d = calculate_3d_metrics(args.root_dir + "/val/Volume")

            current_dice_3d = np.mean(dsc_3d)
            if current_dice_3d > best_dice_3d:
                best_dice_3d = current_dice_3d
                best_epoch = i
                torch.save(net.state_dict(), args.root_dir + "/save/net.pth")

            if i % (best_epoch + 50) == 0:
                for param_group in optimizer.param_groups:
                    lr = lr * 0.5
                    param_group['lr'] = lr

            dice_3d_class = np.mean(dsc_3d, 0)
            val_bar.set_postfix_str(
                "Dice 3D: {:.3f} || Liver: {:.3f}, Kidney(R): {:.3f}, Kidney(L): {:.3f}, Spleen: {:.3f}"
                    .format(np.mean(dice_3d_class), dice_3d_class[0], dice_3d_class[1], dice_3d_class[2],
                            dice_3d_class[3])
            )

            # Save Statistics
            lossG.append(loss_train / len(train_loader))
            dsc.append(dsc_3d)
            assd.append(assd_3d)
            vs.append(vs_3d)

            np.save(args.root_dir + "/save/loss", lossG)
            np.save(args.root_dir + "/save/dsc", dsc)
            np.save(args.root_dir + "/save/assd", assd)
            np.save(args.root_dir + "/save/vs", vs)


def run_eval(args):
    transform = transforms.Compose([transforms.ToTensor()])
    test_chaos_dataset = ChaosDataset(mode="test", root_dir=args.root_dir, transform_input=transform,
                                      transform_mask=GrayToClass(), augment=None)
    test_loader = DataLoader(test_chaos_dataset, batch_size=16, num_workers=args.num_workers, shuffle=False)

    net = MSDualGuided().cuda()
    net.load_state_dict(torch.load(args.checkpoint_path))
    net.eval()

    with tqdm(total=len(test_loader), ascii=True, position=0) as test_bar:
        test_bar.set_description('[Evaluation]')

        for test_image, test_mask, test_img_name in test_loader:
            test_image, test_mask = test_image.cuda(), test_mask.cuda()

            with torch.no_grad():
                seg_pred = net(test_image)
                prediction_to_png(seg_pred, test_img_name, out_path=args.root_dir + "/test/Result")

            test_bar.update()

        create_3d_volume(args.root_dir + "/test/Result", args.root_dir + "/test/Volume/Pred")
        dsc_3d, assd_3d, vs_3d = calculate_3d_metrics(args.root_dir + "/test/Volume")

        test_bar.set_postfix_str(
            "Dice 3D: {:.3f} | ASSD: {:.3f} | VS: {:.3f}"
                .format(np.mean(dsc_3d), np.mean(assd_3d), np.mean(vs_3d))
        )

        np.save(args.root_dir + "/save/dsc_test", dsc_3d)
        np.save(args.root_dir + "/save/assd_test", assd_3d)
        np.save(args.root_dir + "/save/vs_test", vs_3d)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', default='../rawdata/CHAOS_Train_Sets/Train_Sets/MR', type=str)
    parser.add_argument('--root_dir', default='../rawdata/chaos', type=str)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--epochs', default=150, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--create_hierarchy', default=False, action='store_true')
    parser.add_argument('--train', default=False, action='store_true')
    parser.add_argument('--eval', default=False, action='store_true')
    parser.add_argument('--checkpoint_path', default='../rawdata/chaos/save/net.pth', type=str)

    args = parser.parse_args()

    if args.create_hierarchy:
        print("Creating folder for the model!\n")
        shutil.rmtree(args.root_dir, ignore_errors=True)
        create_hierarchy(data_dir=args.data_dir, out_dir=args.root_dir)

    if args.train:
        run_training(args)

    if args.eval:
        run_eval(args)
