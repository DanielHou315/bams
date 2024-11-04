from collections import defaultdict

import numpy as np
import pandas as pd
import seaborn as sn
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.nn as nn

from bams.legged_robots import LeggedRobotsDataset
from bams.models import BAMS, make_byol_predictor
from bams import train_linear_classfier, train_linear_regressor

import matplotlib
matplotlib.use('Agg')


def earth_mover_distance(y_true, y_pred):
    return torch.mean(torch.square(torch.cumsum(y_true, dim=-1) - torch.cumsum(y_pred, dim=-1)), dim=-1)


class WassersteinLoss(nn.Module):
    def __init__(self, skip_frames=60):
        super().__init__()
        # self.loss = SamplesLoss("sinkhorn", p=2, blur=0.1)
        self.skip_frames = skip_frames

    def forward(self, target, pred):
        weights = torch.ones(target.shape[:-1], dtype=torch.float, device=target.device)
        weights[:, :self.skip_frames] = 0.
        weights[:, -53:] = 0.

        target = target.reshape(-1, 32)
        pred = pred.reshape(-1, 32)
        weights = weights.flatten()
        pred = torch.softmax(pred, dim=1)
        loss = earth_mover_distance(target, pred)
        loss = torch.mean(weights * loss)
        return loss


def train(model, device, train_loader, optimizer, short_predictor, long_predictor, epoch):
    model.train()
    criterion = WassersteinLoss(200)
    for batch_idx, data in enumerate(train_loader):
        # move to gpu
        x = data['input'].to(device)
        y = data['target'].to(device)

        # forward
        optimizer.zero_grad()
        pred_x, embs = model(x)

        start_predicting = 200  # skip first 50 samples

        loss = criterion(y.transpose(1, 2), pred_x)

        # contrastive loss: short term
        short_term_emb = embs['short_term']
        batch_size, emb_dim, sequence_length = short_term_emb.size()
        delta = 10
        contrastive_batch_size = 1024
        sample_idx = torch.arange(batch_size).repeat_interleave(contrastive_batch_size//batch_size)
        view_1_id = torch.randint(sequence_length - start_predicting - delta, (contrastive_batch_size,)) + start_predicting
        view_2_id = torch.randint(delta + 1, (contrastive_batch_size,)) + view_1_id
        view_2_id = torch.clip(view_2_id, 0, sequence_length)

        view_1 = short_term_emb[sample_idx, :, view_1_id]
        view_2 = short_term_emb[sample_idx, :, view_2_id]

        q_1 = short_predictor(view_1)
        q_2 = short_predictor(view_2)
        contrastive_loss_1 = 1 - 0.5 * F.cosine_similarity(q_1, view_2.clone().detach(), dim=-1).mean() - \
                             0.5 * F.cosine_similarity(q_2, view_1.clone().detach(), dim=-1).mean()

        # contrastive loss: long term
        long_term_emb = embs['long_term']
        batch_size, emb_dim, sequence_length = long_term_emb.size()
        view_1_id = torch.randint(sequence_length - start_predicting, (contrastive_batch_size,)) + start_predicting
        view_2_id = torch.randint(sequence_length - start_predicting, (contrastive_batch_size,)) + start_predicting

        view_1 = long_term_emb[sample_idx, :, view_1_id]
        view_2 = long_term_emb[sample_idx, :, view_2_id]

        q_1 = long_predictor(view_1)
        q_2 = long_predictor(view_2)
        contrastive_loss_2 = 1 - 0.5 * F.cosine_similarity(q_1, view_2.clone().detach(), dim=-1).mean() - \
                             0.5 * F.cosine_similarity(q_2, view_1.clone().detach(), dim=-1).mean()

        # total loss
        total_loss = loss + 2 * contrastive_loss_1 + contrastive_loss_2

        total_loss.backward()
        optimizer.step()

        if batch_idx % 2 == 0:
            loss = loss.item()
            writer.add_scalar('train/predictive', loss, epoch)
            writer.add_scalar('train/short_term', contrastive_loss_1, epoch)
            writer.add_scalar('train/long_term', contrastive_loss_1, epoch)
            writer.add_scalar('train/total_loss', total_loss, epoch)


def test(model, device, dataset, train_idx, test_idx, writer, epoch):
    # get embeddings
    embeddings = compute_representations(model, dataset, device)

    # decode from all three embeddings
    def decode_class(keys, target, global_pool=False):
        if len(keys) == 1:
            emb = embeddings[keys[0]]
        else:
            emb = torch.cat([embeddings[key] for key in keys], dim=1)
        emb_size = emb.size(1)

        if global_pool:
            emb = torch.mean(emb, dim=-1, keepdim=True)

        train_data = [emb[train_idx].transpose(1, 2).reshape(-1, emb_size), target[train_idx].reshape(-1)]
        test_data = [emb[test_idx].transpose(1, 2).reshape(-1, emb_size), target[test_idx].reshape(-1)]
        f1_score, cm = train_linear_classfier(target.max()+1, train_data, test_data, device, lr=1e-2, weight_decay=1e-4)
        return f1_score, cm

    def decode_scalar(keys, target, global_pool=False):
        if len(keys) == 1:
            emb = embeddings[keys[0]]
        else:
            emb = torch.cat([embeddings[key] for key in keys], dim=1)
        emb_size = emb.size(1)

        if global_pool:
            emb = torch.mean(emb, dim=-1, keepdim=True)

        train_data = [emb[train_idx].transpose(1, 2).reshape(-1, emb_size), target[train_idx].reshape(-1, 1)]
        test_data = [emb[test_idx].transpose(1, 2).reshape(-1, emb_size), target[test_idx].reshape(-1, 1)]
        mse = train_linear_regressor(train_data, test_data, device, lr=1e-2, weight_decay=1e-4)
        return mse

    for emb_keys in [['recent_past', 'short_term', 'long_term']]: # ['recent_past'], ['short_term'], ['long_term']]:
        for target_tag in ['robot_type', 'terrain_type', 'terrain_type_2']:
            target = torch.LongTensor(dataset.data[target_tag])
            if target_tag == 'robot_type':
                class_names = dataset.names['robot_names']
                global_pool = True
            elif target_tag == 'terrain_type':
                class_names = dataset.names['terrain_names']
                global_pool = False
            else:
                class_names = dataset.names['terrain_names_2']
                global_pool = False
            f1_score, cm = decode_class(emb_keys, target, global_pool=global_pool)

            emb_tag = '_'.join(emb_keys)

            writer.add_scalar(f'test/f1_{target_tag}_{emb_tag}', f1_score, epoch)

            writer.add_figure(f'{target_tag}_{emb_tag}',
                              sn.heatmap(pd.DataFrame(cm, index=class_names, columns=class_names), annot=True).get_figure(),
                              epoch)

        for target_tag in ['body_mass', 'command_vel', 'reward', 'terrain_difficulty', 'terrain_slope']:
            target = torch.FloatTensor(dataset.data[target_tag])

            global_pool = target_tag in ['body_mass', 'command_vel']
            mse = decode_scalar(emb_keys, target, global_pool=global_pool)

            emb_tag = '_'.join(emb_keys)
            writer.add_scalar(f'test/mse_{target_tag}_{emb_tag}', mse, epoch)


def compute_representations(model, dataset, device, batch_size=64):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    embs_dict = defaultdict(list)
    for data in loader:
        x = data['input'].to(device)

        with torch.inference_mode():
            pred, embs = model(x)
            for key, emb in embs.items():
                embs_dict[key].append(emb.detach().cpu())

    embs = {key: torch.cat(emb_list) for key, emb_list in embs_dict.items()}
    return embs

# load dataset
dataset = LeggedRobotsDataset('./data')

print("There are {} robots in the dataset.".format(len(dataset)))

# split into train and test
train_idx, test_idx = train_test_split(np.arange(len(dataset)), test_size=0.8, random_state=42)
train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)

print("Number of training robots: {}".format(len(train_idx)))

# prepare dataloaders
batch_size = 64
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size = dataset[0]['input'].shape[0]
target_size = dataset[0]['target'].shape[0]

print("Input size: {}, Target size: {}".format(input_size, target_size))
# predict positions

# fps = 30
model = BAMS(
    input_size=input_size,
    recent_past=dict(num_inputs=input_size, num_channels=(16, 16), kernel_size=2, num_layers_per_block=1),
    short_term=dict(num_inputs=input_size, num_channels=(32, 32, 32), kernel_size=3, num_layers_per_block=2),
    long_term=dict(num_inputs=input_size, num_channels=(32, 32, 32, 32, 32), kernel_size=3, dilation=4, num_layers_per_block=2),
    predictor=dict(hidden_layers=(-1, 64, 128, target_size * 32)),
).to(device)

print(model)

# predictors for latent predictions
short_predictor = make_byol_predictor(32, 256).to(device)
long_predictor = make_byol_predictor(32, 256).to(device)

# optimizer
lr = 1e-3
epochs = 2000
optimizer = optim.AdamW(
    [
        {'params': model.parameters()},
        {'params': short_predictor.parameters(), 'lr': lr * 10},
        {'params': long_predictor.parameters(), 'lr': lr * 10}
    ], lr=lr, weight_decay=1e-5)

scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[500], gamma=0.1)

# writer
writer = SummaryWriter(comment='tcn-robot')

for epoch in tqdm(range(1, epochs + 1)):
    train(model, device, train_loader, optimizer, short_predictor, long_predictor, epoch)
    scheduler.step()
    if epoch % 100 == 0:
        test(model, device, dataset, train_idx, test_idx, writer, epoch)
        torch.save({'model': model}, 'model.pt')
