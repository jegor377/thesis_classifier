import torch
import torch.nn as nn


class LiarPlusStatementsClassifier(nn.Module):
    def __init__(self, encoder_model, inputs, num_hidden, num_classes):
        super(LiarPlusStatementsClassifier, self)
        self.encoder = encoder_model
        self.hl = nn.Linear(
            self.encoder.config.hidden_size * inputs, num_hidden
        )
        self.fc = nn.Linear(num_hidden, num_classes)

    def forward(self, input_ids, attention_mask):
        pass
