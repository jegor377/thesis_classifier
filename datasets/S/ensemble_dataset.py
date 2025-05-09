import pandas as pd
import torch
from torch.utils.data import Dataset

LABEL_MAPPING = {
    "pants-fire": 0,
    "false": 1,
    "barely-true": 2,
    "half-true": 3,
    "mostly-true": 4,
    "true": 5,
}


class LiarPlusStatementsEnsembleDataset(Dataset):
    def __init__(
        self, filepath, tokenizers, device: torch.device, max_length=128
    ):
        self.df = pd.read_csv(filepath, sep="\t")
        self.tokenizers = tokenizers
        self.max_length = max_length
        self.device = device

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, index):
        statement = self.df.iloc[index]["statement"]
        label_str = self.df.iloc[index]["label"]

        # Convert label to integer
        label = LABEL_MAPPING[label_str]

        # Tokenize the statement
        input_ids = []
        attention_mask = []

        for tokenizer in self.tokenizers:
            encoded = tokenizer(
                statement,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {
                k: v.to(self.device).squeeze(0) for k, v in encoded.items()
            }
            input_ids.append(encoded["input_ids"])
            attention_mask.append(encoded["attention_mask"])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": torch.tensor(label, dtype=torch.long).to(
                self.device
            ),  # Ensure tensor
        }
