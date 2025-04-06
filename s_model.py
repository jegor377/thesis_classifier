import torch
import torch.nn as nn


class LiarPlusStatementsClassifier(nn.Module):
    def __init__(self, encoder_model, num_classes):
        super(LiarPlusStatementsClassifier, self).__init__()
        self.encoder = encoder_model  # Pretrained encoder
        self.fc = nn.Linear(self.encoder.config.hidden_size, num_classes)  # Custom classifier

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():  # Ensure encoder remains frozen
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # Use [CLS] token output
        logits = self.fc(cls_output)  # Pass through trainable classifier
        return logits