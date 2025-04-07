import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import ElectraModel, ElectraTokenizer

from checkpoint_utils import load_best_model
from datasets.S.dataset import LiarPlusStatementsDataset
from evaluator import evaluate
from models.S.model import LiarPlusStatementsClassifier

if __name__ == "__main__":
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load ELECTRA tokenizer and model
    tokenizer = ElectraTokenizer.from_pretrained(
        "google/electra-base-discriminator"
    )
    encoder_model = ElectraModel.from_pretrained(
        "google/electra-base-discriminator"
    )
    for param in encoder_model.parameters():
        param.requires_grad = False  # Freeze ELECTRA layers

    # Instantiate your classifier model
    num_classes = 6
    model = LiarPlusStatementsClassifier(encoder_model, num_classes)
    model.to(device)

    # Load the best model (assumes best_model.pth is in the project directory)
    best_model_path = "results/ELECTRA/S/best_model.pth"
    load_best_model(model, best_model_path)

    # Prepare the test dataset and dataloader
    test_dataset = LiarPlusStatementsDataset("data/test2.tsv", tokenizer)
    batch_size = 64
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Evaluate the model
    avg_loss, accuracy = evaluate(model, test_dataloader, criterion, device)
    print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}")
