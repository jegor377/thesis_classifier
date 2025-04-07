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
from datasets.democratic_dataset import LiarPlusStatementsDemocraticDataset
from evaluator import evaluate
from models.democratic_model import DemocraticModelClassifier
from models.s_model import LiarPlusStatementsClassifier
from models.s_model_xlnet import LiarPlusStatementsClassifierXLNet

if __name__ == "__main__":
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    num_classes = 6

    # Load tokenizer and pretrained RoBERTa model
    roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    roberta_model = RobertaModel.from_pretrained("roberta-base")
    for param in roberta_model.parameters():
        param.requires_grad = False  # Freeze RoBERTa layers

    # Instantiate your classifier model
    best_roberta_model = LiarPlusStatementsClassifier(
        roberta_model, num_classes
    )
    best_roberta_model.to(device)

    # Load the best model (assumes best_model.pth is in the project directory)
    roberta_best_model_path = "results/RoBERTa/S/best_model.pth"
    load_best_model(best_roberta_model, roberta_best_model_path)

    # Load tokenizer and pretrained XLM-RoBERTa model
    xlm_roberta_tokenizer = XLMRobertaTokenizer.from_pretrained(
        "xlm-roberta-base"
    )
    xlm_roberta_model = XLMRobertaModel.from_pretrained("xlm-roberta-base")
    for param in xlm_roberta_model.parameters():
        param.requires_grad = False  # Freeze XLM-RoBERTa layers

    # Instantiate your classifier model
    best_xlm_roberta_model = LiarPlusStatementsClassifier(
        xlm_roberta_model, num_classes
    )
    best_xlm_roberta_model.to(device)

    # Load the best model (assumes best_model.pth is in the project directory)
    xlm_roberta_best_model_path = "results/XLMRoBERTa/S/best_model.pth"
    load_best_model(best_xlm_roberta_model, xlm_roberta_best_model_path)

    # Load ELECTRA tokenizer and model
    electra_tokenizer = ElectraTokenizer.from_pretrained(
        "google/electra-base-discriminator"
    )
    electra_model = ElectraModel.from_pretrained(
        "google/electra-base-discriminator"
    )
    for param in electra_model.parameters():
        param.requires_grad = False  # Freeze ELECTRA layers

    # Instantiate your classifier model
    best_electra_model = LiarPlusStatementsClassifier(
        electra_model, num_classes
    )
    best_electra_model.to(device)

    # Load the best model (assumes best_model.pth is in the project directory)
    electra_best_model_path = "results/ELECTRA/S/best_model.pth"
    load_best_model(best_electra_model, electra_best_model_path)

    # Load ERNIE2.0 tokenizer and model
    ernie20_tokenizer = AutoTokenizer.from_pretrained(
        "nghuyong/ernie-2.0-base-en"
    )
    ernie20_model = AutoModel.from_pretrained("nghuyong/ernie-2.0-base-en")
    for param in ernie20_model.parameters():
        param.requires_grad = False  # Freeze ERNIE2.0 layers

    # Instantiate your classifier model
    best_ernie20_model = LiarPlusStatementsClassifier(
        ernie20_model, num_classes
    )
    best_ernie20_model.to(device)

    # Load the best model (assumes best_model.pth is in the project directory)
    ernie20_best_model_path = "results/ERNIE20/S/best_model.pth"
    load_best_model(best_ernie20_model, ernie20_best_model_path)

    # Load encoder tokenizer and model
    xlnet_tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")
    xlnet_model = XLNetModel.from_pretrained("xlnet-base-cased")
    for param in xlnet_model.parameters():
        param.requires_grad = False  # Freeze XLNet layers

    # Instantiate your classifier model
    best_xlnet_model = LiarPlusStatementsClassifierXLNet(
        xlnet_model, num_classes
    )
    best_xlnet_model.to(device)

    # Load the best model (assumes best_model.pth is in the project directory)
    xlnet_best_model_path = "results/XLNet/S/best_model.pth"
    load_best_model(best_xlnet_model, xlnet_best_model_path)

    model = DemocraticModelClassifier(
        [
            # best_roberta_model,
            # best_xlm_roberta_model,
            best_electra_model,
            # best_ernie20_model,
            # best_xlnet_model
        ],
        num_classes,
    )

    # Prepare the test dataset and dataloader
    test_dataset = LiarPlusStatementsDemocraticDataset(
        "data/test2.tsv",
        [
            # roberta_tokenizer,
            # xlm_roberta_tokenizer,
            electra_tokenizer,
            # ernie20_tokenizer,
            # xlnet_tokenizer
        ],
    )
    batch_size = 64
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Evaluate the model
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating"):
            input_ids = torch.stack(batch["input_ids"]).to(device)
            attention_mask = torch.stack(batch["attention_mask"]).to(device)
            labels = batch["label"].to(device).type(torch.LongTensor)

            outputs = (
                model(input_ids, attention_mask)
                .to(device)
                .type(torch.FloatTensor)
            )
            loss = criterion(outputs, labels)
            total_loss += loss.item() * input_ids.size(1)

            preds = torch.argmax(outputs, dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += input_ids.size(1)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}")
