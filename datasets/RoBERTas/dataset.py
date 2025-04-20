import pandas as pd
import torch
from torch.utils.data import Dataset

LABEL_MAPPING = {
    "false": 0,
    "barely-true": 1,
    "half-true": 2,
    "mostly-true": 3,
    "true": 4,
    "pants-fire": 5,
}


class LiarPlusDataset(Dataset):
    def __init__(
        self,
        filepath: str,
        tokenizer,
        columns: list[str],
        max_length: int = 128,
    ):
        self.df = pd.read_csv(filepath, sep="\t")

        self.columns = columns

        for column in self.columns:
            self.df[column] = self.df[column].astype(str)

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, index: int):
        item = self.df.iloc[index]

        input_ids = []
        attention_mask = []

        for column in self.columns:
            encoded = self.tokenizer(
                item[column],
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )
            input_ids.append(encoded["input_ids"])
            attention_mask.append(encoded["attention_mask"])

        input_ids = torch.cat(input_ids, dim=0)
        attention_mask = torch.cat(attention_mask, dim=0)

        label = LABEL_MAPPING[item["label"]]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": torch.tensor(label),
        }
