import os
import torch


def save_checkpoint(model, optimizer, epoch, val_acc, path="checkpoint.pth"):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "val_acc": val_acc
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved at epoch {epoch} with validation accuracy {val_acc:.4f}")

def load_checkpoint(model, optimizer, path="checkpoint.pth", resume=False):
    if os.path.exists(path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"]
        val_acc = checkpoint["val_acc"]
        if resume:
            print(f"Checkpoint loaded: Resuming from epoch {epoch+1}, validation accuracy {val_acc:.4f}")
            return epoch + 1, val_acc  # Next epoch to train
        else:
            print(f"Checkpoint loaded: Starting from first epoch, validation accuracy {val_acc:.4f}")
            return 0, val_acc # Start fresh with existing model
    else:
        print("No checkpoint found. Starting from scratch.")
        return 0, 0 # Start fresh

def load_model(model, path="checkpoint.pth"):
    if os.path.exists(path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Model loaded from checkpoint.")
    else:
        print("No checkpoint found.")