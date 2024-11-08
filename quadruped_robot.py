import os
from tracemalloc import start
import numpy as np
import argparse
from datetime import datetime
import re

import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


from bams.data import KeypointsDataset
from bams.models import BAMS
from bams import HoALoss


from train_utils import save_checkpoint, restore_checkpoint

## Copied from original training script
# def earth_mover_distance(y_true, y_pred):
#     return torch.mean(torch.square(torch.cumsum(y_true, dim=-1) - torch.cumsum(y_pred, dim=-1)), dim=-1)

# class WassersteinLoss(nn.Module):
#     def __init__(self, skip_frames=60):
#         super().__init__()
#         # self.loss = SamplesLoss("sinkhorn", p=2, blur=0.1)
#         self.skip_frames = skip_frames

#     def forward(self, target, pred):
#         weights = torch.ones(target.shape[:-1], dtype=torch.float, device=target.device)
#         weights[:, :self.skip_frames] = 0.
#         weights[:, -53:] = 0.

#         target = target.reshape(-1, 32)
#         pred = pred.reshape(-1, 32)
#         weights = weights.flatten()
#         pred = torch.softmax(pred, dim=1)
#         loss = earth_mover_distance(target, pred)
#         loss = torch.mean(weights * loss)
#         return loss



def load_data(path):
    '''
    Load and format keypoint data. Output should be in the shape (n_samples, seq_len, num_feats). 
    Collapse xy coordinates into the single num_feats dimension.
    '''
    data = np.load(path, allow_pickle=True).item()
    # names contains the class names for each label
    names = data.pop('names')
    # data is a dictionary that contains the behavior timeseries and labels
    state = data['dof_pos']
    action = data['dof_vel']
    # Their dataset comes in the wrong shape. Should be
    # - (num_seq, num_timestamp, num_feature)
    # Theirs came in 
    # - (num_seq, num_feature, num_timestamp)
    out = np.concatenate((state, action), axis=1).transpose(0,2,1)
    print("-----Loaded", path, "as", out.shape, "array")
    print(data.keys())
    return out

def train(model, device, loader, optimizer, criterion, writer, step, log_every_step):
    model.train()

    for data in tqdm(loader, position=1, leave=False):
        # todo convert to float
        input = data["input"].float().to(device)  # (B, N, L)
        target = data["target_hist"].float().to(device)
        ignore_weights = data["ignore_weights"].to(device)

        # forward pass
        optimizer.zero_grad()
        embs, hoa_pred, byol_preds = model(input)

        # prediction task
        hoa_loss = criterion(target, hoa_pred, ignore_weights)

        # contrastive loss: short term
        batch_size, sequence_length, emb_dim = embs["short_term"].size()
        skip_frames, delta = 60, 5
        view_1_id = (
            torch.randint(sequence_length - skip_frames - delta, (batch_size,))
            + skip_frames
        )
        view_2_id = torch.randint(delta + 1, (batch_size,)) + view_1_id
        view_2_id = torch.clip(view_2_id, 0, sequence_length)

        view_1 = byol_preds["short_term"][torch.arange(batch_size), view_1_id]
        view_2 = embs["short_term"][torch.arange(batch_size), view_2_id]

        byol_loss_short_term = (
            1 - F.cosine_similarity(view_1, view_2.clone().detach(), dim=-1).mean()
        )

        # contrastive loss: long term
        batch_size, sequence_length, emb_dim = embs["long_term"].size()
        skip_frames = 100
        view_1_id = (
            torch.randint(sequence_length - skip_frames, (batch_size,)) + skip_frames
        )
        view_2_id = (
            torch.randint(sequence_length - skip_frames, (batch_size,)) + skip_frames
        )

        view_1 = byol_preds["long_term"][torch.arange(batch_size), view_1_id]
        view_2 = embs["long_term"][torch.arange(batch_size), view_2_id]

        byol_loss_long_term = (
            1 - F.cosine_similarity(view_1, view_2.clone().detach(), dim=-1).mean()
        )

        # backprop
        loss = 5e2 * hoa_loss + 0.5 * byol_loss_short_term + 0.5 * byol_loss_long_term

        loss.backward()
        optimizer.step()

        step += 1
        if step % log_every_step == 0:
            writer.add_scalar("train/hoa_loss", hoa_loss.item(), step)
            writer.add_scalar(
                "train/byol_loss_short_term", byol_loss_short_term.item(), step
            )
            writer.add_scalar(
                "train/byol_loss_long_term", byol_loss_long_term.item(), step
            )
            writer.add_scalar("train/total_loss", loss.item(), step)

    return step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./data/mabe")
    parser.add_argument("--cache_path", type=str, default="./data/mabe/custom_dataset")
    parser.add_argument("--hoa_bins", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--log_every_step", type=int, default=50)
    parser.add_argument("--ckpt_file", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset
    keypoints = load_data(args.data_root)

    dataset = KeypointsDataset(
        keypoints=keypoints,
        hoa_bins=args.hoa_bins,
        cache_path=args.cache_path,
        cache=False,
    )
    print("Number of sequences:", len(dataset))

    # prepare dataloaders
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # build model
    # model = BAMS(
    #     input_size=dataset.input_size,
    #     short_term=dict(num_channels=(64, 64, 64, 64), kernel_size=3),
    #     long_term=dict(num_channels=(64, 64, 64, 64, 64), kernel_size=3, dilation=4),
    #     predictor=dict(
    #         hidden_layers=(-1, 256, 512, 512, dataset.target_size * args.hoa_bins)
    #     ),
    # ).to(device)

    # Replaced with original script models
    model = BAMS(
        input_size=dataset.input_size,
        recent_past=dict(num_channels=(16, 16), kernel_size=2, num_layers_per_block=1),
        short_term=dict(num_channels=(32, 32, 32), kernel_size=3, num_layers_per_block=2),
        long_term=dict(num_channels=(32, 32, 32, 32, 32), kernel_size=3, dilation=4, num_layers_per_block=2),
        predictor=dict(
            hidden_layers=(-1, 64, 128, dataset.target_size * args.hoa_bins)
        ),
    ).to(device)

    print(model)
    
    model_name = f"quadruped-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"

    # Load Model
    start_epoch = 1
    if args.ckpt_file is not None:
        model_name, start_epoch, model, scheduler, optimizer = restore_checkpoint(args.ckpt_file, device, model, scheduler, optimizer)
    

    writer = SummaryWriter("runs/" + model_name)

    main_params = [p for name, p in model.named_parameters() if "byol" not in name]
    byol_params = list(model.byol_predictors.parameters())

    optimizer = optim.AdamW(
        [{"params": main_params}, {"params": byol_params, "lr": args.lr * 10}],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200], gamma=0.1)
    criterion = HoALoss(hoa_bins=args.hoa_bins, skip_frames=100)
    # criterion = WassersteinLoss(200)

    step = 0
    for epoch in tqdm(range(start_epoch, args.epochs + 1), position=0):
        step = train(
            model,
            device,
            train_loader,
            optimizer,
            criterion,
            writer,
            step,
            args.log_every_step,
        )
        scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            save_checkpoint(model_name, epoch, model, scheduler, optimizer)


if __name__ == "__main__":
    main()
