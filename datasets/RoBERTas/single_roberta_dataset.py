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


class LiarPlusSingleRobertaDataset(Dataset):
    def __init__(
        self,
        filepath: str,
        tokenizer,
        str_metadata_cols: list[str],
        num_metadata_cols: list[str],
        max_length: int = 512,
    ):
        self.df = pd.read_csv(filepath)

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
