import argparse

import mlflow
import torch
from torch.utils.data import DataLoader
from transformers import RobertaModel, RobertaTokenizer

from datasets.RoBERTas.dataset import LiarPlusDataset
from models.RoBERTas.multiple_robertas import (
    LiarPlusMultipleRoBERTasClassifier,
)
from trainer import train

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
    columns = ["statement", "justification"]

    training_data = LiarPlusDataset("data/train2.tsv", tokenizer, columns)
    validation_data = LiarPlusDataset("data/val2.tsv", tokenizer, columns)

    batch_size = 64

    train_dataloader = DataLoader(
        training_data, batch_size=batch_size, shuffle=True
    )
    val_dataloader = DataLoader(
        validation_data, batch_size=batch_size, shuffle=True
    )

    # Instantiate model
    model = LiarPlusMultipleRoBERTasClassifier(
        roberta, len(columns), hidden_size, num_classes
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
