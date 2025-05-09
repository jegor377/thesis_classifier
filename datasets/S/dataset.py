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


class LiarPlusStatementsDataset(Dataset):
    def __init__(self, filepath: str, tokenizer, max_length: int = 128):
        self.df = pd.read_csv(filepath, sep="\t")
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, index: int):
        statement = self.df.iloc[index]["statement"]
        label_str = self.df.iloc[index]["label"]

        # Convert label to integer
        label = LABEL_MAPPING[label_str]

        # Tokenize the statement
        encoding = self.tokenizer(
            statement,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),  # Remove batch dim
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),  # Ensure tensor
        }
