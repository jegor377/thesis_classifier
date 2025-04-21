import torch
import torch.nn as nn
import torch.nn.functional as F


class LiarPlusMultipleRoBERTasClassifier(nn.Module):
    def __init__(
        self, encoder_model, inputs, num_metadata_len, num_hidden, num_classes
    ):
        super(LiarPlusMultipleRoBERTasClassifier, self).__init__()
        self.encoder = encoder_model
        self.hl = nn.Linear(
            self.encoder.config.hidden_size * inputs + num_metadata_len,
            num_hidden,
        )
        self.fc = nn.Linear(num_hidden, num_classes)

    def forward(self, input_ids, attention_mask, num_metadata):
        batch_size, num_fields, max_length = input_ids.shape

        # reshape from (batch_size, num_fields, max_length) to (batch_size * num_fields, max_length)
        flat_input_ids = input_ids.view(batch_size * num_fields, max_length)
        flat_attention_mask = attention_mask.view(
            batch_size * num_fields, max_length
        )

        with torch.no_grad():  # Ensure encoder remains frozen
            outputs = self.encoder(
                input_ids=flat_input_ids, attention_mask=flat_attention_mask
            )

        # hidden_size should be 768 for RoBERTa
        # shape (batch_size * num_fields, hidden_size)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]

        # reshape (batch_size * num_fields, hidden_size) -> (batch_size, num_fields, hidden_size)
        cls_reshaped = cls_embeddings.view(batch_size, num_fields, -1)

        # reshape (batch_size, num_fields, hidden_size) -> (batch_size, num_fields * hidden_size)
        # which is concatenation along seperate fields' CLS token for following classification
        flattened_cls = torch.flatten(cls_reshaped, start_dim=1)

        concatted_inputs = torch.cat([flattened_cls, num_metadata], dim=1)

        # pass through hidden layer for better feature selection
        hl_output = F.gelu(self.hl(concatted_inputs))

        # pass through classification layer
        logits = self.fc(hl_output)

        return logits
