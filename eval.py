import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaModel
from tqdm import tqdm


from dataset import LiarPlusDataset
from model import LiarPlusClassifier
from checkpoint_utils import load_best_model


def evaluate(model, dataloader, criterion, device):
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * input_ids.size(0)

            preds = torch.argmax(outputs, dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += input_ids.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


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
    model = LiarPlusClassifier(roberta, num_classes)
    model.to(device)

    # Load the best model (assumes best_model.pth is in the project directory)
    best_model_path = "best_model.pth"
    load_best_model(model, best_model_path)

    # Prepare the test dataset and dataloader
    test_dataset = LiarPlusDataset("data/test2.tsv", tokenizer)
    batch_size = 16
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Evaluate the model
    avg_loss, accuracy = evaluate(model, test_dataloader, criterion, device)
    print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}")
