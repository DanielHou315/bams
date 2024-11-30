import os
import torch
import pickle
import numpy as np

def save_checkpoint(model_name, epoch, model, scheduler, optimizer):
    ckpt_path = "checkpoints/"
    ckpt_name = model_name + f"_ep{epoch}" + ".pt"
    ckpt_file = os.path.join(ckpt_path, ckpt_name)
    torch.save({
        'identity': model_name,
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler': scheduler,
    }, ckpt_file)
    
def restore_checkpoint(ckpt_file, device, model, scheduler=None, optimizer=None):
    """
    Restore model from checkpoint if it exists
    Returns the model and the current epoch.
    """
    checkpoint = torch.load(ckpt_file, map_location=device)
    
    model_name = checkpoint["identity"]
    epoch = checkpoint["epoch"] + 1
    model.load_state_dict(checkpoint["model_state_dict"])
    if scheduler is not None:
        scheduler = checkpoint["scheduler"]
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return model_name, epoch, model, scheduler, optimizer

def save_embedding(embs, root_path):
    for key, val in embs.items():
        with open(os.path.join(root_path, key+"_emb.pkl"), 'wb') as f:
            pickle.dump(val, f)
    # print("Embedding saved")

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
    # print(f"Embedding loaded as {rtn.size()}")
    return rtn

def has_embedding_files(keys, root_path):
    for key in keys:
        if not os.path.exists(os.path.join(root_path, key+"_emb.pkl")):
            return False
    return True
