import argparse

import mlflow
import torch
from torch.utils.data import DataLoader
from transformers import XLMRobertaModel, XLMRobertaTokenizer

from datasets.S.dataset import LiarPlusStatementsDataset
from models.S.model import LiarPlusStatementsClassifier
from trainer import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="train.py",
        description="Trains LiarPlusStatementsClassifier with XLM-RoBERTa",
    )

    parser.add_argument("-m", "--mlflow-uri", required=True)
    parser.add_argument("-r", "--resume", action="store_true")
    parser.add_argument("-e", "--reset-epoch", action="store_true")

    args = parser.parse_args()

    mlflow.set_tracking_uri(uri=args.mlflow_uri)

    # MLflow experiment setup
    mlflow.set_experiment("XLM-RoBERTa_LiarPlus_Classification")

    # Load RoBERTa tokenizer and model
    tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
    roberta = XLMRobertaModel.from_pretrained("xlm-roberta-base")

    for param in roberta.parameters():
        param.requires_grad = False  # Freeze all layers

    training_data = LiarPlusStatementsDataset("data/train2.tsv", tokenizer)
    validation_data = LiarPlusStatementsDataset("data/val2.tsv", tokenizer)

    batch_size = 64

    train_dataloader = DataLoader(
        training_data, batch_size=batch_size, shuffle=True
    )
    val_dataloader = DataLoader(
        validation_data, batch_size=batch_size, shuffle=True
    )

    # Hyperparameters
    num_classes = 6
    lr = 1e-3
    epochs = 30

    # Instantiate model
    model = LiarPlusStatementsClassifier(roberta, num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train(
        model,
        "results/S/XLMRoBERTa",
        train_dataloader,
        val_dataloader,
        batch_size,
        lr,
        epochs,
        args.resume,
        args.reset_epoch,
    )
