import os
import torch


def save_checkpoint(model, optimizer, epoch, loss, path="checkpoint.pth"):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "loss": loss
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved at epoch {epoch} with loss {loss:.4f}")

def load_checkpoint(model, optimizer, path="checkpoint.pth", resume=False):
    if os.path.exists(path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"]
        loss = checkpoint["loss"]
        if resume:
            print(f"Checkpoint loaded: Resuming from epoch {epoch+1}, loss {loss:.4f}")
            return epoch + 1, loss  # Next epoch to train
        else:
            print(f"Checkpoint loaded: Starting from first epoch, loss {loss:.4f}")
            return 0, loss # Start fresh with existing model
    else:
        print("No checkpoint found. Starting from scratch.")
        return 0, float("inf")  # Start fresh

def load_model(model, path="checkpoint.pth"):
    if os.path.exists(path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Model loaded from checkpoint.")
    else:
        print("No checkpoint found.")