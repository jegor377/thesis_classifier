import torch
import torch.nn as nn
import torch.nn.functional as F


class DemocraticModelClassifier(nn.Module):
    def __init__(self, classifiers, num_classes):
        super(DemocraticModelClassifier, self).__init__()
        self.classifiers = classifiers
        self.num_classes = num_classes

    def forward(self, input_ids, attention_mask):
        # (classifiers_count, input_size)
        outputs = torch.zeros((len(input_ids), input_ids[0].shape[0]))

        for i in range(len(self.classifiers)):
            classifier = self.classifiers[i]
            logits = classifier(input_ids[i], attention_mask[i])
            preds = torch.argmax(logits, dim=1)
            outputs[i] = preds

        results = torch.mode(outputs, dim=0).values.to(torch.int64)

        return torch.nn.functional.one_hot(results, self.num_classes)
