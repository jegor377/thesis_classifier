import torch
import mlflow
import argparse

from transformers import XLNetTokenizer, XLNetModel
from torch.utils.data import DataLoader

from datasets.dataset import LiarPlusStatementsDataset
from models.s_model_xlnet import LiarPlusStatementsClassifierXLNet
from trainer import train


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='train.py',
        description='Trains LiarPlusStatementsClassifier with XLNet')

    parser.add_argument('-m', '--mlflow-uri', required=True)
    parser.add_argument('-r', '--resume',
                        action='store_true')
    parser.add_argument('-e', '--reset-epoch',
                        action='store_true')
    
    args = parser.parse_args()
    
    mlflow.set_tracking_uri(uri=args.mlflow_uri)
    
    # MLflow experiment setup
    mlflow.set_experiment("XLNet_LiarPlus_Classification")
    
    # Load encoder tokenizer and model
    tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")
    encoder_model = XLNetModel.from_pretrained("xlnet-base-cased")
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
    model = LiarPlusStatementsClassifierXLNet(encoder_model, num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    train(
        model,
        'results/XLNet/S',
        train_dataloader,
        val_dataloader,
        batch_size,
        lr,
        epochs,
        args.resume,
        args.reset_epoch
    )
