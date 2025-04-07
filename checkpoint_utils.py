import os

import torch


def save_checkpoint(model, optimizer, epoch, val_acc, path="checkpoint.pth"):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "val_acc": val_acc,
    }
    torch.save(checkpoint, path)
    print(
        f"Checkpoint saved at epoch {epoch} "
        f"with validation accuracy {val_acc:.4f}"
    )


def load_checkpoint(
    model, optimizer, path="checkpoint.pth", resume=False, reset_epoch=False
):
    if not resume:
        print("Resume is False. Starting from scratch.")
        return 0, 0  # Start fresh

    if os.path.exists(path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"]
        val_acc = checkpoint["val_acc"]
        if reset_epoch:
            print(
                f"Checkpoint loaded: Starting from initial"
                f"epoch, validation accuracy {val_acc:.4f}"
            )
            return 0, val_acc  # Start fresh with existing model
        else:
            print(
                f"Checkpoint loaded: Resuming from epoch "
                f"{epoch+1}, validation accuracy {val_acc:.4f}"
            )
            return epoch + 1, val_acc  # Next epoch to train
    else:
        print("No checkpoint found. Starting from scratch.")
        return 0, 0  # Start fresh


def save_best_model(model, optimizer, epoch, val_acc, path="best_model.pth"):
    best_model = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "val_acc": val_acc,
    }
    torch.save(best_model, path)
    print(
        f"Best model saved at epoch {epoch} "
        f"with validation accuracy {val_acc:.4f}"
    )


def load_best_model(model, path="best_model.pth"):
    if os.path.exists(path):
        best_model = torch.load(path)
        model.load_state_dict(best_model["model_state_dict"])
        print("Model loaded from best model checkpoint.")
    else:
        print("No best model checkpoint found.")
