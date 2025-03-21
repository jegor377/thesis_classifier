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

def load_checkpoint(model, optimizer, path="checkpoint.pth"):
    if os.path.exists(path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"]
        loss = checkpoint["loss"]
        print(f"Checkpoint loaded: Resuming from epoch {epoch+1}, loss {loss:.4f}")
        return epoch + 1  # Next epoch to train
    else:
        print("No checkpoint found. Starting from scratch.")
        return 0  # Start fresh