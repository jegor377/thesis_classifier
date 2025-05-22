import argparse

import mlflow
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import RobertaModel, RobertaTokenizer
from tqdm import tqdm
import time
from torchmetrics.classification import (
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)

from datasets.RoBERTas.single_roberta_dataset_subset import (
    LiarPlusSingleRobertaDatasetSubset,
    ids2labels,
)
from models.RoBERTas.single_roberta import LiarPlusSingleRoBERTasClassifier

from utils import load_checkpoint, save_best_model, save_checkpoint


def train(
    model: nn.Module,
    save_path: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    batch_size: int,
    num_classes: int,
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

        f1 = MulticlassF1Score(num_classes, average=None).to(device)
        precision = MulticlassPrecision(num_classes, average=None).to(device)
        recall = MulticlassRecall(num_classes, average=None).to(device)

        # Training loop
        for epoch in range(start_epoch, epochs):
            model.train()
            epoch_loss = 0

            train_accuracy = 0

            f1.reset()
            precision.reset()
            recall.reset()

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

                f1.update(preds, labels)
                precision.update(preds, labels)
                recall.update(preds, labels)

            avg_loss = epoch_loss / len(train_loader)
            avg_train_accuracy = train_accuracy / len(train_loader.dataset)
            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            mlflow.log_metric("train_acc", avg_train_accuracy, step=epoch)

            f1_res = f1.compute()
            precision_res = precision.compute()
            recall_res = recall.compute()

            for i in range(num_classes):
                mlflow.log_metric(
                    f"train_f1_{ids2labels[i]}", f1_res[i], step=epoch
                )
                mlflow.log_metric(
                    f"train_precision_{ids2labels[i]}",
                    precision_res[i],
                    step=epoch,
                )
                mlflow.log_metric(
                    f"train_recall_{ids2labels[i]}", recall_res[i], step=epoch
                )

            macro_f1 = f1_res.mean()
            macro_precision = precision_res.mean()
            macro_recall = recall_res.mean()

            mlflow.log_metric("train_f1", macro_f1, step=epoch)
            mlflow.log_metric("train_precision", macro_precision, step=epoch)
            mlflow.log_metric("train_recall", macro_recall, step=epoch)

            tqdm.write(
                f"Epoch {epoch+1}: "
                f"Training Loss: {avg_loss}, "
                f"Training Accuracy: {avg_train_accuracy}, "
                f"Training F1: {macro_f1}, "
                f"Training Precision: {macro_precision}, "
                f"Training Recall: {macro_recall}"
            )

            # Validation step
            model.eval()  # Switch to evaluation mode
            val_loss = 0
            val_accuracy = 0

            f1.reset()
            precision.reset()
            recall.reset()

            with torch.no_grad():
                for batch in tqdm(
                    val_loader,
                    desc=f"Validation of epoch {epoch + 1}",
                    leave=False,
                ):
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    num_metadata = batch["num_metadata"].to(device)
                    labels = batch["label"].to(device)

                    outputs = model(input_ids, attention_mask, num_metadata)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

                    # Calculate accuracy
                    preds = torch.argmax(outputs, dim=-1)
                    val_accuracy += (preds == labels).sum().item()
                    f1.update(preds, labels)
                    precision.update(preds, labels)
                    recall.update(preds, labels)

            avg_val_loss = val_loss / len(val_loader)
            avg_val_accuracy = val_accuracy / len(val_loader.dataset)
            mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", avg_val_accuracy, step=epoch)

            f1_res = f1.compute()
            precision_res = precision.compute()
            recall_res = recall.compute()

            for i in range(num_classes):
                mlflow.log_metric(
                    f"val_f1_{ids2labels[i]}", f1_res[i], step=epoch
                )
                mlflow.log_metric(
                    f"val_precision_{ids2labels[i]}",
                    precision_res[i],
                    step=epoch,
                )
                mlflow.log_metric(
                    f"val_recall_{ids2labels[i]}", recall_res[i], step=epoch
                )

            macro_f1 = f1_res.mean()
            macro_precision = precision_res.mean()
            macro_recall = recall_res.mean()

            mlflow.log_metric("val_f1", macro_f1, step=epoch)
            mlflow.log_metric("val_precision", macro_precision, step=epoch)
            mlflow.log_metric("val_recall", macro_recall, step=epoch)

            print(
                f"Epoch {epoch+1}: "
                f"Validation Loss: {avg_val_loss}, "
                f"Validation Accuracy: {avg_val_accuracy}, "
                f"Validation F1: {macro_f1}, "
                f"Validation Precision: {macro_precision}, "
                f"Validation Recall: {macro_recall}"
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
        description="Trains LiarPlusSingleRoBERTasClassifier",
    )

    parser.add_argument("-m", "--mlflow-uri", required=True)
    parser.add_argument("-r", "--resume", action="store_true")
    parser.add_argument("-e", "--reset-epoch", action="store_true")

    args = parser.parse_args()

    mlflow.set_tracking_uri(uri=args.mlflow_uri)

    # MLflow experiment setup
    mlflow.set_experiment("LiarPlusSingleRoBERTasClassifier")

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
        "subject",
        "speaker",
        "job_title",
        "state",
        "party_affiliation",
        "context",
    ]
    num_metadata_cols = [
        "barely_true_counts",
        "false_counts",
        "half_true_counts",
        "mostly_true_counts",
        "pants_on_fire_counts",
    ]

    subset_size = 1000
    random_state = 42

    training_data_subset = LiarPlusSingleRobertaDatasetSubset(
        subset_size,
        "data/normalized/train2.csv",
        tokenizer,
        text_columns,
        num_metadata_cols,
        random_state,
    )
    validation_data = LiarPlusSingleRobertaDatasetSubset(
        -1,
        "data/normalized/val2.csv",
        tokenizer,
        text_columns,
        num_metadata_cols,
    )

    batch_size = 64

    train_dataloader = DataLoader(
        training_data_subset, batch_size=batch_size, shuffle=True
    )
    val_dataloader = DataLoader(
        validation_data, batch_size=batch_size, shuffle=True
    )

    # Instantiate model
    model = LiarPlusSingleRoBERTasClassifier(
        roberta,
        len(num_metadata_cols),
        hidden_size,
        num_classes,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    start = time.time()
    train(
        model,
        "results/RoBERTas/single_roberta",
        train_dataloader,
        val_dataloader,
        batch_size,
        num_classes,
        lr,
        epochs,
        args.resume,
        args.reset_epoch,
    )
    end = time.time()
    print(f"Total time took training: {end-start}s")
