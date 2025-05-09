import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.utils import resample

LABEL_MAPPING = {
    "pants-fire": 0,
    "false": 1,
    "barely-true": 2,
    "half-true": 3,
    "mostly-true": 4,
    "true": 5,
}

ids2labels = [
    "pants-fire",
    "false",
    "barely-true",
    "half-true",
    "mostly-true",
    "true",
]


class LiarPlusDatasetSubset(Dataset):
    def __init__(
        self,
        total_size: int,
        filepath: str,
        tokenizer,
        columns: list[str],
        num_metadata_cols: list[str],
        random_state: int | None = None,
        max_length: int = 128,
    ):
        num_classes = 6
        df = pd.read_csv(filepath)

        if total_size != -1:
            desired_count = total_size // num_classes

            self.df = pd.concat(
                [
                    resample(
                        group,
                        replace=False,
                        n_samples=desired_count,
                        random_state=random_state,
                    )
                    for _, group in df.groupby("label")
                ]
            )
        else:
            self.df = df

        self.columns = columns
        self.num_metadata_cols = num_metadata_cols

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

        metadata = [item[column] for column in self.num_metadata_cols]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "num_metadata": torch.tensor(metadata).float(),
            "label": torch.tensor(label),
        }
