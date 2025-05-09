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


class LiarPlusSingleRobertaDatasetSubset(Dataset):
    def __init__(
        self,
        total_size: int,
        filepath: str,
        tokenizer,
        str_metadata_cols: list[str],
        num_metadata_cols: list[str],
        random_state: int | None = None,
        max_length: int = 512,
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

        self.str_metadata_cols = str_metadata_cols
        self.num_metadata_cols = num_metadata_cols

        for column in self.str_metadata_cols:
            self.df[column] = self.df[column].astype(str)

        self.df["statement"] = self.df["statement"].astype(str)
        self.df["justification"] = self.df["justification"].astype(str)

        self.statement_max_len = max_length // 4
        self.justification_max_len = max_length // 4
        self.str_metadata_max_len = (
            max_length - self.statement_max_len - self.justification_max_len
        ) // len(str_metadata_cols)

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df.index)

    def limit_tokens(self, text, max_length=512):
        tokenized = self.tokenizer.tokenize(text)[:max_length]
        return self.tokenizer.decode(
            self.tokenizer.convert_tokens_to_ids(tokenized)
        )

    def __getitem__(self, index: int):
        item = self.df.iloc[index]

        input = self.limit_tokens(
            f"[STATEMENT] {item['statement']} ", self.statement_max_len
        )
        input += self.limit_tokens(
            f" [JUSTIFICATION] {item['justification']}",
            self.justification_max_len,
        )

        for column in self.str_metadata_cols:
            input += self.limit_tokens(
                f" [{column.upper()}] {item[column]}",
                self.str_metadata_max_len,
            )

        encoded = self.tokenizer(
            input,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        label = LABEL_MAPPING[item["label"]]

        num_metadata = [item[column] for column in self.num_metadata_cols]

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "num_metadata": torch.tensor(num_metadata).float(),
            "label": torch.tensor(label),
        }
