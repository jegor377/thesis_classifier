import torch
import torch.nn as nn


class EnsembleModelClassifier(nn.Module):
    def __init__(self, classifiers: list[nn.Module], num_classes: int=6):
        super(EnsembleModelClassifier, self).__init__()
        self.classifiers = classifiers
        for model in self.classifiers:
            model.eval()
        self.num_classes = num_classes

    def forward(self, input_ids: list[torch.tensor],
                attention_mask: list[torch.tensor]) -> torch.tensor:
        """Forward method of the model.

        Args:
            input_ids (list[torch.tensor]): list of input ids tensors from LiarPlusStatementsEnsembleDataset
            attention_mask (list[torch.tensor]): list of attention mask tensors from LiarPlusStatementsEnsembleDataset

        Returns:
            torch.tensor: Ensemble model output logits.
        """
        outputs = []
        
        for i, model in enumerate(self.classifiers):
            with torch.no_grad():
                out = model(input_ids[i], attention_mask[i])
                outputs.append(out)
        
        # Average outputs
        avg_output = torch.mean(torch.stack(outputs), dim=0)
        return avg_output
