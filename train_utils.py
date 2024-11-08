import os
import torch

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

