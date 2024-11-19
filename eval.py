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


def main():
    # load in checkpoint
    dataset = #TODO
    train_idx, test_idx = train_test_split(np.arrange(len(dataset)), test_size=0.8, random_state=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = BAMS(
        input_size=dataset.input_size,
        recent_past=dict(num_channels=(16, 16), kernel_size=2, num_layers_per_block=1),
        short_term=dict(num_channels=(32, 32, 32), kernel_size=3, num_layers_per_block=2),
        long_term=dict(num_channels=(32, 32, 32, 32, 32), kernel_size=3, dilation=4, num_layers_per_block=2),
        predictor=dict(
            hidden_laye
            rs=(-1, 64, 128, dataset.target_size * args.hoa_bins)
        ),
    ).to(device)
    writer=SummaryWriter("runs/" + model_name)

    test(model, device, dataset, train_idx, test_idx, writer)
    torch.save({'model': model}, 'bams.pt')
if __name__ == "__main__":
    main()