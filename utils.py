import os
import torch
import paramiko  # type: ignore
from tqdm import tqdm


LABEL_MAPPING = {
    "pants-fire": 0,
    "false": 1,
    "barely-true": 2,
    "half-true": 3,
    "mostly-true": 4,
    "true": 5,
}

ids2labels = [
    "pants-fire",
    "false",
    "barely-true",
    "half-true",
    "mostly-true",
    "true",
]


def save_checkpoint(model, optimizer, epoch, val_acc, path="checkpoint.pth"):
    checkpoint = {
        "model_state_dict": model.state_for_save(),
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
        model.load_state_from_save(checkpoint["model_state_dict"])
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
        "model_state_dict": model.state_for_save(),
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
        model.load_state_from_save(best_model["model_state_dict"])
        print("Model loaded from best model checkpoint.")
    else:
        print("No best model checkpoint found.")


def save_model_remotely(local_path, remote_path, creds):
    # Ustawienia SSH
    hostname = creds["hostname"]
    port = creds["port"]
    username = creds["username"]
    password = creds["password"]

    # Połączenie SSH
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(hostname, port=port, username=username, password=password)

        # Pobierz rozmiar pliku lokalnego
        file_size = os.path.getsize(local_path)

        # Funkcja do aktualizacji paska postępu
        def progress_callback(transferred, total):
            progress_bar.update(transferred - progress_bar.n)

        # Inicjalizuj pasek postępu
        progress_bar = tqdm(
            total=file_size,
            unit="B",
            unit_scale=True,
            desc=f"Uploading {local_path}",
        )

        # SFTP transfer z callbackiem
        with ssh.open_sftp() as sftp:
            temp_remote_path = (
                remote_path + os.path.basename(local_path) + ".tmp"
            )
            final_remote_path = remote_path + os.path.basename(local_path)

            sftp.put(local_path, temp_remote_path, callback=progress_callback)

            try:
                sftp.remove(final_remote_path)
            except IOError:
                # Plik nie istnieje – można ignorować
                pass

            sftp.rename(temp_remote_path, final_remote_path)

        # Po zakończeniu
        progress_bar.close()
        print(f"Plik {os.path.basename(local_path)} został wysłany.")

    except Exception as e:
        print(f"Error: {e}")

    finally:
        # Zapewnia, że połączenie SSH zawsze zostanie zamknięte
        ssh.close()
