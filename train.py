import torch
import torch.nn as nn
import mlflow
import mlflow.pytorch
import argparse

from transformers import RobertaModel, RobertaTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import LiarPlusDataset
from model import LiarPlusClassifier
from checkpoint_utils import (save_checkpoint,
                              load_checkpoint,
                              save_best_model)


def train(train_loader: DataLoader,
          val_loader: DataLoader,
          batch_size: int,
          resume: bool=False,
          reset_epoch: bool=False) -> None:
    with mlflow.start_run():
        # Hyperparameters
        num_classes = 6
        lr = 1e-3
        epochs = 30
        
        mlflow.log_param("learning_rate", lr)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("resume", resume)
        mlflow.log_param("reset_epoch", reset_epoch)

        # Instantiate model
        model = LiarPlusClassifier(roberta, num_classes)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Define optimizer and loss function
        # Train only the classifier
        optimizer = torch.optim.Adam(model.fc.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        # Checkpoint Path
        checkpoint_path = "checkpoint.pth"
        # Best model path
        best_model_path = "best_model.pth"
        
        # Track best loss for model saving
        # Load Checkpoint (Decide if you want to continue)
        start_epoch, best_val_accuracy = load_checkpoint(model,
                                      optimizer,
                                      checkpoint_path,
                                      resume,
                                      reset_epoch)
        
        # Early stopping and validation-based checkpointing
        patience = 5  # Number of epochs to wait before stopping if no improvement
        patience_counter = 0
        
        mlflow.log_param("patience", 5)

        # Training loop
        for epoch in range(start_epoch, epochs):
            model.train()
            epoch_loss = 0
            
            train_accuracy = 0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
                # Calculate accuracy
                preds = torch.argmax(outputs, dim=-1)
                train_accuracy += (preds == labels).sum().item()
            
            avg_loss = epoch_loss / len(train_loader)
            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            mlflow.log_metric("train_acc", train_accuracy, step=epoch)
            
            tqdm.write(f"Epoch {epoch+1}, Training accuracy: {train_accuracy}, Training loss: {avg_loss}")
            
            # Validation step
            model.eval()  # Switch to evaluation mode
            val_loss = 0
            val_accuracy = 0
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Validation of epoch {epoch + 1}", leave=False):
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

            print(f"Epoch {epoch+1}, Validation Loss: {avg_val_loss}, Validation Accuracy: {avg_val_accuracy}")
            
            save_checkpoint(model, optimizer, epoch, avg_val_loss, checkpoint_path)
            
            # Check for early stopping
            if avg_val_accuracy > best_val_accuracy:
                best_val_accuracy = avg_val_accuracy
                patience_counter = 0
                # Save the best model
                save_best_model(model, optimizer, epoch, avg_val_loss, best_model_path)
                mlflow.log_artifact(best_model_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            
        # Log final model
        mlflow.pytorch.log_model(model, "classifier_model")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='train.py',
        description='Trains LiarPlusClassifier')

    parser.add_argument('-m', '--mlflow-uri', required=True)
    parser.add_argument('-r', '--resume',
                        action='store_true')
    parser.add_argument('-e', '--reset-epoch',
                        action='store_true')
    
    args = parser.parse_args()
    
    mlflow.set_tracking_uri(uri=args.mlflow_uri)
    
    # MLflow experiment setup
    mlflow.set_experiment("RoBERTa_LiarPlus_Classification")
    
    # Load RoBERTa tokenizer and model
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    roberta = RobertaModel.from_pretrained("roberta-base")
    
    for param in roberta.parameters():
        param.requires_grad = False  # Freeze all layers
    
    training_data = LiarPlusDataset("data/train2.tsv", tokenizer)
    validation_data = LiarPlusDataset("data/val2.tsv", tokenizer)
    
    batch_size = 16
    
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(validation_data, batch_size=batch_size, shuffle=True)
    
    train(train_dataloader, val_dataloader, batch_size, args.resume, args.reset_epoch)
