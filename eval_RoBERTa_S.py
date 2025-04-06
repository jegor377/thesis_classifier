import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaModel


from dataset import LiarPlusStatementsDataset
from s_model import LiarPlusStatementsClassifier
from checkpoint_utils import load_best_model

from evaluator import evaluate


if __name__ == '__main__':
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer and pretrained RoBERTa model
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    roberta = RobertaModel.from_pretrained("roberta-base")
    for param in roberta.parameters():
        param.requires_grad = False  # Freeze RoBERTa layers

    # Instantiate your classifier model
    num_classes = 6
    model = LiarPlusStatementsClassifier(roberta, num_classes)
    model.to(device)

    # Load the best model (assumes best_model.pth is in the project directory)
    best_model_path = "models/RoBERTa/S/best_model.pth"
    load_best_model(model, best_model_path)

    # Prepare the test dataset and dataloader
    test_dataset = LiarPlusStatementsDataset("data/test2.tsv", tokenizer)
    batch_size = 64
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Evaluate the model
    avg_loss, accuracy = evaluate(model, test_dataloader, criterion, device)
    print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}")
