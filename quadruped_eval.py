from collections import defaultdict
import linecache
import os
import tracemalloc
from sys import getsizeof
import pickle

import numpy as np
import pandas as pd
import argparse
import seaborn as sn
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

from bams.data import KeypointsDataset
# from torch.utils.data import Dataset
from bams.models import BAMS
from bams import train_linear_classfier, train_linear_regressor

from train_utils import save_checkpoint, restore_checkpoint

import matplotlib
matplotlib.use('Agg')


class LeggedRobotDataset(KeypointsDataset):
    def __init__(self, data_file):
        '''
        Load and format keypoint data. Output should be in the shape (n_samples, seq_len, num_feats). 
        Collapse xy coordinates into the single num_feats dimension.
        '''
        # NOTE: 
        # Their dataset comes in the wrong shape. Should be
        # - (num_seq, num_timestamp, num_feature)
        # Theirs came in 
        # - (num_seq, num_feature, num_timestamp)
        # HOWEVER, for this script, we will stick to the original shapes 
        # as it works with their original script
        data = np.load(data_file, allow_pickle=True).item()
        # names contains the class names for each label
        self.names = data.pop('names')
        # Put everything else as self.data
        self.data = data

        # Make data for dataset
        input_feats = np.concatenate((data["dof_pos"], data["dof_vel"]), axis=1).transpose(0,2,1)

        super().__init__(input_feats)

def save_embedding(embs, root_path):
    for key, val in embs.items():
        with open(os.path.join(root_path, key+"_emb.pkl"), 'wb') as f:
            pickle.dump(val, f)
    print("Embedding saved")

def load_embedding(keys, root_path):
    vals = []
    for key in keys:
        file = os.path.join(root_path, key+"_emb.pkl")
        if not os.path.isfile(file):
            raise Exception(f"Embedding with key={key} not found!")
        with open(os.path.join(root_path, key+"_emb.pkl"), 'rb') as f:
            val = pickle.load(f)
            vals.append(val)

    if len(vals) == 1:
        rtn = vals[0]
    rtn = torch.cat(vals, dim=1)
    print(f"Embedding loaded as {rtn.size()}")
    return rtn

def has_embedding_files(keys, root_path):
    for key in keys:
        if not os.path.exists(os.path.join(root_path, key+"_emb.pkl")):
            return False
    return True


def test(model, device, dataset, train_idx, test_idx, writer, epoch):
    # get embeddings
    # embeddings = compute_representations(model, dataset, device, cache_file=f"./cache/quadruped_ep{epoch}_embs.npy")
    cache_path = f"./cache/quadruped_ep{epoch}_embs"
    compute_representations(model, dataset, device, cache_path, \
        keys=['recent_past', 'short_term', 'long_term'])

    # decode from all three embeddings
    def decode_class(keys, cache_dir, target, global_pool=False):
        # if len(keys) == 1:
        #     emb = embeddings[keys[0]]
        # else:
        #     emb = torch.cat([embeddings[key] for key in keys], dim=1)
        emb = load_embedding(keys, cache_dir)
        emb_size = emb.size(1)

        if global_pool:
            emb = torch.mean(emb, dim=-1, keepdim=True)

        train_data = [emb[train_idx].transpose(1, 2).reshape(-1, emb_size), target[train_idx].reshape(-1)]
        test_data = [emb[test_idx].transpose(1, 2).reshape(-1, emb_size), target[test_idx].reshape(-1)]
        f1_score, cm = train_linear_classfier(target.max()+1, train_data, test_data, device, lr=1e-2, weight_decay=1e-4)
        del emb
        return f1_score, cm

    def decode_scalar(keys, cache_dir, target, global_pool=False):
        # if len(keys) == 1:
        #     emb = embeddings[keys[0]]
        # else:
        #     emb = torch.cat([embeddings[key] for key in keys], dim=1)
        emb = load_embedding(keys, cache_dir)
        emb_size = emb.size(1)

        if global_pool:
            emb = torch.mean(emb, dim=-1, keepdim=True)

        train_data = [emb[train_idx].transpose(1, 2).reshape(-1, emb_size), target[train_idx].reshape(-1, 1)]
        test_data = [emb[test_idx].transpose(1, 2).reshape(-1, emb_size), target[test_idx].reshape(-1, 1)]
        mse = train_linear_regressor(train_data, test_data, device, lr=1e-2, weight_decay=1e-4)
        del emb
        return mse

    del dataset.input_feats, dataset.target_feats

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

            f1_score, cm = decode_class(emb_keys, cache_path, target, global_pool=global_pool)
            emb_tag = '_'.join(emb_keys)
            writer.add_scalar(f'test/f1_{target_tag}_{emb_tag}', f1_score, epoch)
            writer.add_figure(f'{target_tag}_{emb_tag}',
                              sn.heatmap(pd.DataFrame(cm, index=class_names, columns=class_names), annot=True).get_figure(),
                              epoch)
            print("F-1 score == {:.3f} for {} with\n  -- Embeddings {}".format(f1_score, target_tag, emb_keys))

        for target_tag in ['body_mass', 'command_vel', 'reward', 'terrain_difficulty', 'terrain_slope']:
            target = torch.FloatTensor(dataset.data[target_tag])

            global_pool = target_tag in ['body_mass', 'command_vel']
            mse = decode_scalar(emb_keys, cache_path, target, global_pool=global_pool)

            emb_tag = '_'.join(emb_keys)
            writer.add_scalar(f'test/mse_{target_tag}_{emb_tag}', mse, epoch)
            print("-- MSE == {:.3f} for {} with\n  -- Embeddings {}".format(mse, target_tag, emb_keys))

def compute_representations(model, dataset, device, cache_path, keys, batch_size=64):
    """
    dataset: input data
    embs: dictionary of embeddings with name of TCN as key and 
            output shape: nSeries x nTimeStamp x nEmbedding
    """
    if os.path.exists(cache_path) and has_embedding_files(keys, cache_path):
        return
    os.makedirs(cache_path, exist_ok=True)

    # Otherwise, compute
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    embs_dict = defaultdict(list)
    for data in tqdm(loader):
        x = data['input'].to(device)

        with torch.no_grad():
            embs, hoa_pred, byol_pred = model(x)
            # WE ARE SUPPOSED TO CHANGE THE SHAPE BACK, which is annoying but necessary to make other parts of this script work.
            for key, emb in embs.items():
                embs_dict[key].append(emb.transpose(2,1).detach().cpu())

    embs = {key: torch.cat(emb_list) for key, emb_list in embs_dict.items()}
    save_embedding(embs, cache_path)
    del embs_dict, embs

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

    tracemalloc.start()

    dataset = LeggedRobotDataset(args.data_root)
    train_idx, test_idx = train_test_split(np.arange(len(dataset)), test_size=0.8, random_state=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Dataset size: {getsizeof(dataset)}")

    model = BAMS(
        input_size=dataset.input_size,
        recent_past=dict(num_channels=(16, 16), kernel_size=2, num_layers_per_block=1),
        short_term=dict(num_channels=(32, 32, 32), kernel_size=3, num_layers_per_block=2),
        long_term=dict(num_channels=(32, 32, 32, 32, 32), kernel_size=3, dilation=4, num_layers_per_block=2),
        predictor=dict(
            hidden_layers=(-1, 64, 128, dataset.target_size * args.hoa_bins)
        ),
    ).to(device)

    # restore model 
    model_name, epoch, model, _, _ = restore_checkpoint(args.ckpt_file, device, model)
    writer=SummaryWriter("runs/" + model_name)

    test(model, device, dataset, train_idx, test_idx, writer, epoch)

if __name__ == "__main__":
    main()