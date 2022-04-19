import os
import os.path as osp
import time
import math
from datetime import timedelta
from argparse import ArgumentParser

import torch
from torch import cuda
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm

from east_dataset import EASTDataset
from dataset import SceneTextDataset
from model import EAST

import wandb

wandb.init(project="data-annotation", entity="medic", name = "ICDAR19+ICDAR17+epoch400+BATCH12_with_valid")


def fix_seed() :
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(0)

def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.environ.get("SM_CHANNEL_TRAIN", "../input/data/datasets/ko_en"),
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=os.environ.get("SM_MODEL_DIR", "trained_models"),
    )

    parser.add_argument("--device", default="cuda" if cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--image_size", type=int, default=1024)
    parser.add_argument("--input_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--max_epoch", type=int, default=400)
    parser.add_argument("--save_interval", type=int, default=5)

    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError("`input_size` must be a multiple of 32")

    return args


def do_training(
    data_dir,
    model_dir,
    device,
    image_size,
    input_size,
    num_workers,
    batch_size,
    learning_rate,
    max_epoch,
    save_interval,
):
    train_dataset = SceneTextDataset(
        data_dir, split="train", image_size=image_size, crop_size=input_size
    )
    train_dataset = EASTDataset(train_dataset)
    num_batches_train = math.ceil(len(train_dataset) / batch_size)

    valid_dataset = SceneTextDataset(
        data_dir,
        split="valid",
        image_size=image_size,
        crop_size=input_size,
        train=False,
    )
    valid_dataset = EASTDataset(valid_dataset)
    num_batches_val = math.ceil(len(valid_dataset) / batch_size)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=True,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST()
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0.001)

    model.train()
    for epoch in range(max_epoch):
        epoch_loss_train, epoch_start = 0, time.time()
        with tqdm(total=num_batches_train) as pbar:
            for img, gt_score_map, gt_geo_map, roi_mask in train_loader:
                pbar.set_description("[Epoch {}]".format(epoch + 1))

                train_loss, extra_info_train = model.train_step(
                    img, gt_score_map, gt_geo_map, roi_mask
                )
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                loss_val = train_loss.item()
                epoch_loss_train += loss_val

                pbar.update(1)
                val_dict = {
                    "Cls loss": extra_info_train["cls_loss"],
                    "Angle loss": extra_info_train["angle_loss"],
                    "IoU loss": extra_info_train["iou_loss"],
                }
                pbar.set_postfix(val_dict)
        print(
            "Mean loss: {:.4f} | Elapsed time: {}".format(
                epoch_loss_train / num_batches_train,
                timedelta(seconds=time.time() - epoch_start),
            )
        )
        model.eval()
        with torch.no_grad():
            epoch_loss_val, epoch_start = 0, time.time()
            with tqdm(total=num_batches_val) as pbar:
                for img, gt_score_map, gt_geo_map, roi_mask in val_loader:
                    pbar.set_description("[Epoch {}]".format(epoch + 1))

                    val_loss, extra_info_val = model.train_step(
                        img, gt_score_map, gt_geo_map, roi_mask
                    )

                    loss_val = val_loss.item()
                    epoch_loss_val += loss_val

                    pbar.update(1)
                    val_dict = {
                        "Cls loss": extra_info_val["cls_loss"],
                        "Angle loss": extra_info_val["angle_loss"],
                        "IoU loss": extra_info_val["iou_loss"],
                    }
                    pbar.set_postfix(val_dict)
        print(
            "Mean loss: {:.4f} | Elapsed time: {}".format(
                epoch_loss_val / num_batches_val,
                timedelta(seconds=time.time() - epoch_start),
            )
        )
        scheduler.step()

        wandb.log(
            {
                # "train_cls_loss": extra_info_train["cls_loss"],
                # "train_angle_loss": extra_info_train["angle_loss"],
                # "train_iou_loss": extra_info_train["iou_loss"],
                "train_mean_loss": epoch_loss_train / num_batches_train,
                # "val_cls_loss": extra_info_val["cls_loss"],
                # "val_angle_loss": extra_info_val["angle_loss"],
                # "val_iou_loss": extra_info_val["iou_loss"],
                "val_mean_loss": epoch_loss_val / num_batches_val,
            }
        )

        if (epoch + 1) % save_interval == 0:
            if not osp.exists(model_dir):
                os.makedirs(model_dir)

            ckpt_fpath = osp.join(model_dir, "latest.pth")
            torch.save(model.state_dict(), ckpt_fpath)


def main(args):
    fix_seed()
    do_training(**args.__dict__)


if __name__ == "__main__":
    args = parse_args()
    wandb.config.update(args)
    main(args)
