import torch
import mlflow
import argparse

from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader

from datasets.dataset import LiarPlusStatementsDataset
from models.s_model import LiarPlusStatementsClassifier
from trainer import train


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='train.py',
        description='Trains LiarPlusStatementsClassifier with ERNIE 2.0')

    parser.add_argument('-m', '--mlflow-uri', required=True)
    parser.add_argument('-r', '--resume',
                        action='store_true')
    parser.add_argument('-e', '--reset-epoch',
                        action='store_true')
    
    args = parser.parse_args()
    
    mlflow.set_tracking_uri(uri=args.mlflow_uri)
    
    # MLflow experiment setup
    mlflow.set_experiment("ERNIE2.0_LiarPlus_Classification")
    
    # Load encoder tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("nghuyong/ernie-2.0-base-en")
    encoder_model = AutoModel.from_pretrained("nghuyong/ernie-2.0-base-en")
    for param in encoder_model.parameters():
        param.requires_grad = False  # Freeze all layers
    
    training_data = LiarPlusStatementsDataset("data/train2.tsv", tokenizer)
    validation_data = LiarPlusStatementsDataset("data/val2.tsv", tokenizer)
    
    batch_size = 64
    
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(validation_data, batch_size=batch_size, shuffle=True)
    
    # Hyperparameters
    num_classes = 6
    lr = 1e-3
    epochs = 30
    
    # Instantiate model
    model = LiarPlusStatementsClassifier(encoder_model, num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    train(
        model,
        'results/ERNIE20/S',
        train_dataloader,
        val_dataloader,
        batch_size,
        lr,
        epochs,
        args.resume,
        args.reset_epoch
    )
