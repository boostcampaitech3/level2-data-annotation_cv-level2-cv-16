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
import nni
from importlib import import_module
from nni.utils import merge_parameter

wandb.init(project="data-annotation", entity="medic", name="Kyoungmin_nni-experiment-")


def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=os.environ.get("SM_CHANNEL_TRAIN", "../input/data/ICDAR17_Korean"),
    )
    parser.add_argument(
        "--valid_data_dir",
        type=str,
        default=os.environ.get("SM_CHANNEL_VALID", "../input/data/ICDAR17_Korean"),
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
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_epoch", type=int, default=200)
    parser.add_argument("--save_interval", type=int, default=5)
    parser.add_argument("--optimizer", type=str, default="Adam")

    args = parser.parse_args()

    # NNI (Auto-ML) 사용을 위한 argument 재지정
    tuner_params = nni.get_next_parameter()
    args = merge_parameter(args, tuner_params)

    if args.input_size % 32 != 0:
        raise ValueError("`input_size` must be a multiple of 32")

    return args


def do_training(
    train_data_dir,
    valid_data_dir,
    model_dir,
    device,
    image_size,
    input_size,
    num_workers,
    batch_size,
    learning_rate,
    max_epoch,
    save_interval,
    optimizer,
):
    train_dataset = SceneTextDataset(
        train_data_dir, split="train", image_size=image_size, crop_size=input_size
    )
    train_dataset = EASTDataset(train_dataset)
    num_batches = math.ceil(len(train_dataset) / batch_size)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    valid_dataset = SceneTextDataset(
        valid_data_dir, split="val", image_size=image_size, crop_size=input_size
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST()
    model.to(device)
    opt_module = getattr(import_module("torch.optim"), args.optimizer)
    optimizer = opt_module(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=5e-4,
    )
    scheduler = lr_scheduler.MultiStepLR(
        optimizer, milestones=[max_epoch // 2], gamma=0.1
    )

    for epoch in range(max_epoch):
        epoch_loss, epoch_start = 0, time.time()
        model.train()

        with tqdm(total=num_batches) as pbar:
            for img, gt_score_map, gt_geo_map, roi_mask in train_loader:
                pbar.set_description("(TRAIN): [Epoch {}]".format(epoch + 1))

                loss, extra_info = model.train_step(
                    img, gt_score_map, gt_geo_map, roi_mask
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_val = loss.item()
                epoch_loss += loss_val

                pbar.update(1)
                val_dict = {
                    "Cls loss": extra_info["cls_loss"],
                    "Angle loss": extra_info["angle_loss"],
                    "IoU loss": extra_info["iou_loss"],
                }
                pbar.set_postfix(val_dict)

        scheduler.step()
   
        print('Mean loss: {:.4f} | Elapsed time: {}'.format(
            epoch_loss / num_batches, timedelta(seconds=time.time() - epoch_start)))
        
        wandb.log({
            'cls_loss': extra_info['cls_loss'],
            'angle_loss': extra_info['angle_loss'],
            'iou_loss': extra_info['iou_loss'],
            "mean_loss": epoch_loss / num_batches,
        })
      
        nni.report_final_result(epoch_loss / num_batches)

        if (epoch + 1) % save_interval == 0:
            if not osp.exists(model_dir):
                os.makedirs(model_dir)

            ckpt_fpath = osp.join(model_dir, "latest.pth")
            torch.save(model.state_dict(), ckpt_fpath)


def main(args):
    do_training(**args.__dict__)


if __name__ == "__main__":
    args = parse_args()
    wandb.config.update(args)
    main(args)
