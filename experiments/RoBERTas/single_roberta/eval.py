import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import RobertaModel, RobertaTokenizer
from tqdm import tqdm
from torchmetrics.classification import (
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)

from utils import load_best_model
from datasets.RoBERTas.single_roberta_dataset import (
    LiarPlusSingleRobertaDataset,
)
from models.RoBERTas.single_roberta import LiarPlusSingleRoBERTasClassifier


def evaluate(model, dataloader, criterion, device):
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    f1 = MulticlassF1Score(num_classes, average=None).to(device)
    precision = MulticlassPrecision(num_classes, average=None).to(device)
    recall = MulticlassRecall(num_classes, average=None).to(device)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            num_metadata = batch["num_metadata"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask, num_metadata)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * input_ids.size(0)

            preds = torch.argmax(outputs, dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += input_ids.size(0)

            f1.update(preds, labels)
            precision.update(preds, labels)
            recall.update(preds, labels)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    f1_res = f1.compute()
    precision_res = precision.compute()
    recall_res = recall.compute()

    return avg_loss, accuracy, f1_res, precision_res, recall_res


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
    model = LiarPlusSingleRoBERTasClassifier(
        roberta,
        len(num_metadata_cols),
        hidden_size,
        num_classes,
    )
    model.to(device)

    # Load the best model (assumes best_model.pth is in the project directory)
    best_model_path = "results/RoBERTas/single_roberta/best_model.pth"
    load_best_model(model, best_model_path)

    # Prepare the test dataset and dataloader
    test_dataset = LiarPlusSingleRobertaDataset(
        "data/normalized/test2.csv", tokenizer, text_columns, num_metadata_cols
    )
    val_dataset = LiarPlusSingleRobertaDataset(
        "data/normalized/val2.csv", tokenizer, text_columns, num_metadata_cols
    )

    batch_size = 64
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Evaluate the model
    avg_loss, accuracy, f1_res, precision_res, recall_res = evaluate(
        model, test_dataloader, criterion, device
    )

    print(
        f"Test Loss: {avg_loss:.4f}, "
        f"Test Accuracy: {accuracy:.4f}, "
        f"Test F1: {f1_res} (marcro = {f1_res.mean():.4f}), "
        f"Test Precision: {precision_res} (marcro = {precision_res.mean():.4f}), "
        f"Test Recall: {recall_res} (marcro = {recall_res.mean():.4f}), "
    )

    avg_loss, accuracy, f1_res, precision_res, recall_res = evaluate(
        model, val_dataloader, criterion, device
    )

    print(
        f"Val Loss: {avg_loss:.4f}, "
        f"Val Accuracy: {accuracy:.4f}, "
        f"Val F1: {f1_res} (marcro = {f1_res.mean():.4f}), "
        f"Val Precision: {precision_res} (marcro = {precision_res.mean():.4f}), "
        f"Val Recall: {recall_res} (marcro = {recall_res.mean():.4f}), "
    )
