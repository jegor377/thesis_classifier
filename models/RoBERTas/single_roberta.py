import torch
import torch.nn as nn
import torch.nn.functional as F


class LiarPlusSingleRoBERTasClassifier(nn.Module):
    def __init__(
        self, encoder_model, num_metadata_len, num_hidden, num_classes
    ):
        super(LiarPlusSingleRoBERTasClassifier, self).__init__()
        self.encoder = encoder_model
        self.hl = nn.Linear(
            self.encoder.config.hidden_size + num_metadata_len, num_hidden
        )
        self.fc = nn.Linear(num_hidden, num_classes)

    def forward(self, input_ids, attention_mask, num_metadata):
        with torch.no_grad():
            outputs = self.encoder(
                input_ids=input_ids, attention_mask=attention_mask
            )

        cls_embedding = outputs.last_hidden_state[:, 0, :]
        concatted_inputs = torch.cat([cls_embedding, num_metadata], dim=1)

        hl_output = F.gelu(self.hl(concatted_inputs))

        logits = self.fc(hl_output)
        return logits
