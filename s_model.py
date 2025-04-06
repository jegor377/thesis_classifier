import torch
import torch.nn as nn


class LiarPlusStatementsClassifier(nn.Module):
    def __init__(self, roberta_model, num_classes):
        super(LiarPlusStatementsClassifier, self).__init__()
        self.roberta = roberta_model  # Pretrained RoBERTa
        self.fc = nn.Linear(self.roberta.config.hidden_size, num_classes)  # Custom classifier

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():  # Ensure RoBERTa remains frozen
            outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # Use [CLS] token output
        logits = self.fc(cls_output)  # Pass through trainable classifier
        return logits