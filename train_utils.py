import os
import torch

def save_checkpoint(model_name, epoch, model, optimizer, scheduler):
    ckpt_path = "checkpoints/"
    ckpt_name = model_name + f"_ep{epoch}" + ".pt"
    PATH = os.path.join(ckpt_path, ckpt_name)
    torch.save({
        'identity': model_name,
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler': scheduler,
    }, PATH)
    
def restore_checkpoint(ckpt_file, device, model, scheduler, optimizer):
    """
    Restore model from checkpoint if it exists
    Returns the model and the current epoch.
    """
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    epoch = checkpoint["epoch"] + 1
    model.load_state_dict(checkpoint["model_state_dict"])
    scheduler = checkpoint["scheduler"]
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return epoch, model, scheduler, optimizer

