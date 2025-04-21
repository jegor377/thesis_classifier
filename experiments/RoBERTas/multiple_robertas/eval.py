import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import RobertaModel, RobertaTokenizer

from checkpoint_utils import load_best_model
from datasets.RoBERTas.dataset import LiarPlusDataset
from evaluator import evaluate
from models.RoBERTas.multiple_robertas import (
    LiarPlusMultipleRoBERTasClassifier,
)

if __name__ == "__main__":
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer and pretrained RoBERTa model
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    roberta = RobertaModel.from_pretrained("roberta-base")
    for param in roberta.parameters():
        param.requires_grad = False  # Freeze RoBERTa layers

    # Instantiate your classifier model
    num_classes = 6
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
    model = LiarPlusMultipleRoBERTasClassifier(
        roberta,
        len(text_columns),
        len(num_metadata_cols),
        hidden_size,
        num_classes,
    )
    model.to(device)

    # Load the best model (assumes best_model.pth is in the project directory)
    best_model_path = "results/RoBERTas/multiple_robertas/best_model.pth"
    load_best_model(model, best_model_path)

    # Prepare the test dataset and dataloader
    test_dataset = LiarPlusDataset(
        "data/test2.tsv", tokenizer, text_columns, num_metadata_cols
    )
    batch_size = 64
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Evaluate the model
    avg_loss, accuracy = evaluate(model, test_dataloader, criterion, device)
    print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}")
