import pandas as pd
import torch

from torch.utils.data import Dataset


LABEL_MAPPING = {
    "false": 0,
    "barely-true": 1,
    "half-true": 2,
    "mostly-true": 3,
    "true": 4,
    "pants-fire": 5
}


class LiarPlusStatementsDemocraticDataset(Dataset):
    def __init__(self, filepath, tokenizers, max_length=128):
        self.df = pd.read_csv(filepath, sep='\t')
        self.tokenizers = tokenizers
        self.max_length = max_length
        
    
    def __len__(self):
        return len(self.df.index)
    
    def __getitem__(self, index):
        statement = self.df.iloc[index]['statement']
        label_str = self.df.iloc[index]['label']
        
        # Convert label to integer
        label = LABEL_MAPPING[label_str]

        # Tokenize the statement
        encodings = []
        
        for tokenizer in self.tokenizers:
            encodings.append(tokenizer(
                statement,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt"
            ))
            
        input_ids = [encoding["input_ids"].squeeze(0) for encoding in encodings]
        attention_mask = [encoding["attention_mask"].squeeze(0) for encoding in encodings]

        return {
            "input_ids": input_ids,  # Remove batch dim
            "attention_mask": attention_mask,
            "label": torch.tensor(label, dtype=torch.long)  # Ensure tensor
        }