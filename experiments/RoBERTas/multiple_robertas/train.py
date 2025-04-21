import argparse

import mlflow
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import RobertaModel, RobertaTokenizer
from tqdm import tqdm

from datasets.RoBERTas.dataset import LiarPlusDataset
from models.RoBERTas.multiple_robertas import (
    LiarPlusMultipleRoBERTasClassifier,
)

from checkpoint_utils import load_checkpoint, save_best_model, save_checkpoint


def train(
    model: nn.Module,
    save_path: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    batch_size: int,
    lr=1e-3,
    epochs=30,
    resume: bool = False,
    reset_epoch: bool = False,
) -> None:
    with mlflow.start_run():

        mlflow.log_param("learning_rate", lr)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("resume", resume)
        mlflow.log_param("reset_epoch", reset_epoch)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Define optimizer and loss function
        # Train only the classifier
        optimizer = torch.optim.Adam(model.fc.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        # Checkpoint Path
        checkpoint_path = "checkpoint.pth"
        # Best model path
        best_model_path = f"{save_path}/best_model.pth"

        # Track best loss for model saving
        # Load Checkpoint (Decide if you want to continue)
        start_epoch, best_val_accuracy = load_checkpoint(
            model, optimizer, checkpoint_path, resume, reset_epoch
        )

        # Early stopping and validation-based checkpointing
        patience = (
            5  # Number of epochs to wait before stopping if no improvement
        )
        patience_counter = 0

        mlflow.log_param("patience", 5)

        # Training loop
        for epoch in range(start_epoch, epochs):
            model.train()
            epoch_loss = 0

            train_accuracy = 0

            for batch in tqdm(
                train_loader, desc=f"Epoch {epoch+1}", leave=False
            ):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                num_metadata = batch["num_metadata"].to(device)
                labels = batch["label"].to(device)

                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask, num_metadata)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

                # Calculate accuracy
                preds = torch.argmax(outputs, dim=-1)
                train_accuracy += (preds == labels).sum().item()

            avg_loss = epoch_loss / len(train_loader)
            avg_train_accuracy = train_accuracy / len(train_loader.dataset)
            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            mlflow.log_metric("train_acc", avg_train_accuracy, step=epoch)

            tqdm.write(
                f"Epoch {epoch+1}, Training Loss: {avg_loss}, Training Accuracy: {avg_train_accuracy}"
            )

            # Validation step
            model.eval()  # Switch to evaluation mode
            val_loss = 0
            val_accuracy = 0
            with torch.no_grad():
                for batch in tqdm(
                    val_loader,
                    desc=f"Validation of epoch {epoch + 1}",
                    leave=False,
                ):
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["label"].to(device)

                    outputs = model(input_ids, attention_mask)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

                    # Calculate accuracy
                    preds = torch.argmax(outputs, dim=-1)
                    val_accuracy += (preds == labels).sum().item()

            avg_val_loss = val_loss / len(val_loader)
            avg_val_accuracy = val_accuracy / len(val_loader.dataset)
            mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", avg_val_accuracy, step=epoch)

            print(
                f"Epoch {epoch+1}, Validation Loss: {avg_val_loss}, Validation Accuracy: {avg_val_accuracy}"
            )

            save_checkpoint(
                model, optimizer, epoch, avg_val_accuracy, checkpoint_path
            )

            # Check for early stopping
            if avg_val_accuracy > best_val_accuracy:
                best_val_accuracy = avg_val_accuracy
                patience_counter = 0
                # Save the best model
                save_best_model(
                    model, optimizer, epoch, best_val_accuracy, best_model_path
                )
                mlflow.log_artifact(best_model_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        # Log final model
        mlflow.pytorch.log_model(model, "classifier_model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="train.py",
        description="Trains LiarPlusMultipleRoBERTasClassifier",
    )

    parser.add_argument("-m", "--mlflow-uri", required=True)
    parser.add_argument("-r", "--resume", action="store_true")
    parser.add_argument("-e", "--reset-epoch", action="store_true")

    args = parser.parse_args()

    mlflow.set_tracking_uri(uri=args.mlflow_uri)

    # MLflow experiment setup
    mlflow.set_experiment("RoBERTa_LiarPlus_Classification")

    # Load RoBERTa tokenizer and model
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    roberta = RobertaModel.from_pretrained("roberta-base")

    for param in roberta.parameters():
        param.requires_grad = False  # Freeze all layers

    # Hyperparameters
    num_classes = 6
    lr = 1e-3
    epochs = 30
    hidden_size = 128
    text_columns = [
        "statement",
        "subject",
        "speaker",
        "job_title",
        "state",
        "party_affiliation",
        "context",
        "justification",
    ]
    num_metadata_cols = [
        "barely_true_counts",
        "false_counts",
        "half_true_counts",
        "mostly_true_counts",
        "pants_on_fire_counts",
    ]

    training_data = LiarPlusDataset(
        "data/train2.tsv", tokenizer, text_columns, num_metadata_cols
    )
    validation_data = LiarPlusDataset(
        "data/val2.tsv", tokenizer, text_columns, num_metadata_cols
    )

    batch_size = 64

    train_dataloader = DataLoader(
        training_data, batch_size=batch_size, shuffle=True
    )
    val_dataloader = DataLoader(
        validation_data, batch_size=batch_size, shuffle=True
    )

    # Instantiate model
    model = LiarPlusMultipleRoBERTasClassifier(
        roberta,
        len(text_columns),
        len(num_metadata_cols),
        hidden_size,
        num_classes,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train(
        model,
        "results/RoBERTas/multiple_robertas",
        train_dataloader,
        val_dataloader,
        batch_size,
        lr,
        epochs,
        args.resume,
        args.reset_epoch,
    )
