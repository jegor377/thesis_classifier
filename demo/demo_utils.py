import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

examples = [
    [
        "Building a wall on the U.S.-Mexico border will take literally years.",
        "immigration",
        "rick-perry",
        "Governor",
        "Texas",
        "republican",
        "Radio interview",
    ],
    [
        "Wisconsin is on pace to double the number of layoffs this year.",
        "jobs",
        "katrina-shankland",
        "State representative",
        "Wisconsin",
        "democrat",
        "a news conference",
    ],
    [
        "Says John McCain has done nothing to help the vets.",
        "military,veterans,voting-record",
        "donald-trump",
        "President-Elect",
        "New York",
        "republican",
        "comments on ABC's This Week.",
    ],
]

states = [
    "Alabama",
    "Alaska",
    "Arizona",
    "Arkansas",
    "California",
    "Colorado",
    "Connecticut",
    "Delaware",
    "Florida",
    "Georgia",
    "Hawaii",
    "Idaho",
    "Illinois",
    "Indiana",
    "Iowa",
    "Kansas",
    "Kentucky",
    "Louisiana",
    "Maine",
    "Maryland",
    "Massachusetts",
    "Michigan",
    "Minnesota",
    "Mississippi",
    "Missouri",
    "Montana",
    "Nebraska",
    "Nevada",
    "New Hampshire",
    "New Jersey",
    "New Mexico",
    "New York",
    "North Carolina",
    "North Dakota",
    "Ohio",
    "Oklahoma",
    "Oregon",
    "Pennsylvania",
    "Rhode Island",
    "South Carolina",
    "South Dakota",
    "Tennessee",
    "Texas",
    "Utah",
    "Vermont",
    "Virginia",
    "Washington",
    "West Virginia",
    "Wisconsin",
    "Wyoming",
]

normalized_df = pd.read_csv("data/normalized/test2.csv")

num_metadata_cols = [
    "barely_true_counts",
    "false_counts",
    "half_true_counts",
    "mostly_true_counts",
    "pants_on_fire_counts",
    "grammar_errors",
    "ratio_of_capital_letters",
]

one_hot_cols = [
    "sentiment",
    "question",
    "curse",
    "emotion",
    "gibberish",
    "offensiveness",
    "political_bias",
]

one_hot_labels = {
    "sentiment": ["negative", "neutral", "positive"],
    "question": ["not_question", "question"],
    "curse": ["curse", "non-curse"],
    "emotion": [
        "anger",
        "disgust",
        "fear",
        "joy",
        "neutral",
        "sadness",
        "surprise",
    ],
    "gibberish": ["clean", "mild gibberish", "word salad"],
    "offensiveness": ["non-offensive", "offensive"],
    "political_bias": ["CENTER", "LEFT", "RIGHT"],
}

label_to_index = {
    "sentiment": {
        label: idx for idx, label in enumerate(one_hot_labels["sentiment"])
    },
    "question": {
        label: idx for idx, label in enumerate(one_hot_labels["question"])
    },
    "curse": {label: idx for idx, label in enumerate(one_hot_labels["curse"])},
    "emotion": {
        label: idx for idx, label in enumerate(one_hot_labels["emotion"])
    },
    "gibberish": {
        label: idx for idx, label in enumerate(one_hot_labels["gibberish"])
    },
    "offensiveness": {
        label: idx for idx, label in enumerate(one_hot_labels["offensiveness"])
    },
    "political_bias": {
        label: idx
        for idx, label in enumerate(one_hot_labels["political_bias"])
    },
}

one_hot_metadata_size = sum([len(x) for x in one_hot_labels.values()])

statement_to_num_metadata = {
    row["statement"]: [row[col] for col in num_metadata_cols]
    for _, row in normalized_df.iterrows()
}

speaker_to_num_metadata = {
    row["speaker"]: [row[col] for col in num_metadata_cols]
    for _, row in normalized_df.iterrows()
}

ids2labels = [
    "pants-fire",
    "false",
    "barely-true",
    "half-true",
    "mostly-true",
    "true",
]


def limit_tokens(tokenizer, text, max_length=512):
    return tokenizer.convert_tokens_to_string(
        tokenizer.tokenize(text)[:max_length]
    )


class LiarPlusSingleFinetunedRoBERTasClassifier(nn.Module):
    def __init__(
        self,
        encoder_model,
        num_metadata_len,
        one_hot_metadata_size,
        num_hidden,
        num_classes,
    ):
        super(LiarPlusSingleFinetunedRoBERTasClassifier, self).__init__()
        self.encoder = encoder_model
        self.hl = nn.Linear(
            self.encoder.config.hidden_size
            + num_metadata_len
            + one_hot_metadata_size,
            num_hidden,
        )
        self.dropout = nn.Dropout(p=0.1)
        self.fc = nn.Linear(num_hidden, num_classes)

    def forward(
        self, input_ids, attention_mask, num_metadata, one_hot_metadata
    ):
        outputs = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        )

        cls_embedding = outputs.pooler_output
        concatted_inputs = torch.cat(
            [cls_embedding, num_metadata, one_hot_metadata], dim=1
        )

        hl_output = F.gelu(self.hl(concatted_inputs))
        hl_output = self.dropout(hl_output)

        logits = self.fc(hl_output)
        return logits

    def roberta_trainable_state(self):
        return {
            name: param
            for name, param in self.encoder.named_parameters()
            if param.requires_grad
        }

    def load_roberta_trainable_state(self, state_dict):
        self.encoder.load_state_dict(state_dict, strict=False)

    # Zapisz tylko wagi warstw klasyfikatora
    def state_for_save(self):
        return {
            "hl_state_dict": self.hl.state_dict(),
            "fc_state_dict": self.fc.state_dict(),
            "roberta_trainable": self.roberta_trainable_state(),
        }

    # Ładowanie modelu (tylko wagi klasyfikatora)
    def load_state_from_save(self, state):
        self.hl.load_state_dict(state["hl_state_dict"])
        self.fc.load_state_dict(state["fc_state_dict"])
        if "roberta_trainable" in state:
            self.load_roberta_trainable_state(state["roberta_trainable"])
