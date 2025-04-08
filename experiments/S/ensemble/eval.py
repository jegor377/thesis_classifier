import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoTokenizer,
    ElectraModel,
    ElectraTokenizer,
    RobertaModel,
    RobertaTokenizer,
    XLMRobertaModel,
    XLMRobertaTokenizer,
    XLNetModel,
    XLNetTokenizer,
)

from checkpoint_utils import load_best_model
from datasets.S.ensemble_dataset import LiarPlusStatementsEnsembleDataset
from models.S.ensemble_model import EnsembleModelClassifier
from models.S.model import LiarPlusStatementsClassifier
from models.S.xlnet_model import LiarPlusStatementsClassifierXLNet


def load_model(
    name: str,
    best_model_path: str,
    encoder_tokenizer,
    encoder_model,
    classifier,
    num_classes: int,
    device: torch.device,
) -> tuple:
    enc_tokenizer = encoder_tokenizer.from_pretrained(name)
    enc_model = encoder_model.from_pretrained(name)
    for param in enc_model.parameters():
        param.requires_grad = False

    # Instantiate your classifier model
    best_model = classifier(enc_model, num_classes)
    best_model.to(device)

    # Load the best model (assumes best_model.pth is in the project directory)
    load_best_model(best_model, best_model_path)

    return enc_tokenizer, best_model


def evaluate(
    model: nn.Module, dataloader: DataLoader, criterion
) -> tuple[float, float]:
    # Evaluate the model
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["label"]

            outputs = model(input_ids, attention_mask)

            loss = criterion(outputs, labels)
            total_loss += loss.item() * input_ids[0].size(0)

            preds = torch.argmax(outputs, dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += input_ids[0].size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


if __name__ == "__main__":
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    num_classes = 6

    roberta_tokenizer, best_roberta_model = load_model(
        "roberta-base",
        "results/RoBERTa/S/best_model.pth",
        RobertaTokenizer,
        RobertaModel,
        LiarPlusStatementsClassifier,
        num_classes,
        device,
    )

    xlm_roberta_tokenizer, best_xlm_roberta_model = load_model(
        "xlm-roberta-base",
        "results/XLMRoBERTa/S/best_model.pth",
        XLMRobertaTokenizer,
        XLMRobertaModel,
        LiarPlusStatementsClassifier,
        num_classes,
        device,
    )

    electra_tokenizer, best_electra_model = load_model(
        "google/electra-base-discriminator",
        "results/ELECTRA/S/best_model.pth",
        ElectraTokenizer,
        ElectraModel,
        LiarPlusStatementsClassifier,
        num_classes,
        device,
    )

    ernie20_tokenizer, best_ernie20_model = load_model(
        "nghuyong/ernie-2.0-base-en",
        "results/ERNIE20/S/best_model.pth",
        AutoTokenizer,
        AutoModel,
        LiarPlusStatementsClassifier,
        num_classes,
        device,
    )

    xlnet_tokenizer, best_xlnet_model = load_model(
        "xlnet-base-cased",
        "results/XLNet/S/best_model.pth",
        XLNetTokenizer,
        XLNetModel,
        LiarPlusStatementsClassifierXLNet,
        num_classes,
        device,
    )

    model = EnsembleModelClassifier(
        [
            best_roberta_model,
            # best_xlm_roberta_model,
            best_electra_model,
            best_ernie20_model,
            # best_xlnet_model
        ],
        num_classes,
    )

    # Prepare the test dataset and dataloader
    test_dataset = LiarPlusStatementsEnsembleDataset(
        "data/test2.tsv",
        [
            roberta_tokenizer,
            # xlm_roberta_tokenizer,
            electra_tokenizer,
            ernie20_tokenizer,
            # xlnet_tokenizer
        ],
        device,
    )

    batch_size = 64
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    avg_loss, accuracy = evaluate(model, test_dataloader, criterion)

    print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}")
