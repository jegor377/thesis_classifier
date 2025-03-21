import torch
import torch.nn as nn
import mlflow
import mlflow.pytorch

from transformers import RobertaModel, RobertaTokenizer
from torch.utils.data import DataLoader

from dataset import LiarPlusDataset
from model import LiarPlusClassifier
from checkpoint_utils import (save_checkpoint, load_checkpoint)


def train(train_loader: DataLoader, batch_size: int) -> None:
    with mlflow.start_run():
        # Hyperparameters
        num_classes = 6
        lr = 1e-3
        epochs = 3
        
        mlflow.log_param("learning_rate", lr)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("epochs", epochs)

        # Instantiate model
        model = LiarPlusClassifier(roberta, num_classes)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Define optimizer and loss function
        optimizer = torch.optim.Adam(model.fc.parameters(), lr=lr)  # Train only the classifier
        criterion = nn.CrossEntropyLoss()
        
        # Checkpoint Path
        checkpoint_path = "checkpoint.pth"
        
        best_loss = float("inf")  # Track best loss for model saving

        # Load Checkpoint (Decide if you want to continue)
        start_epoch = load_checkpoint(model, optimizer, checkpoint_path)

        # Training loop
        for epoch in range(start_epoch, epochs):
            model.train()
            epoch_loss = 0
            
            for batch in train_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            mlflow.log_metric("loss", avg_loss, step=epoch)
            
            print(f"Epoch {epoch+1}, Loss: {avg_loss}")
            
            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                save_checkpoint(model, optimizer, epoch, avg_loss, checkpoint_path)
                mlflow.log_artifact(checkpoint_path)
        
        # Log final model
        mlflow.pytorch.log_model(model, "classifier_model")


if __name__ == '__main__':
    # MLflow experiment setup
    mlflow.set_experiment("RoBERTa_LiarPlus_Classification")
    
    # Load RoBERTa tokenizer and model
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    roberta = RobertaModel.from_pretrained("roberta-base")
    
    for param in roberta.parameters():
        param.requires_grad = False  # Freeze all layers
    
    training_data = LiarPlusDataset("data/train2.tsv")
    
    batch_size = 16
    
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    
    train(train_dataloader, batch_size)
