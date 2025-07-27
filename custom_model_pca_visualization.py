import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel
from sklearn.decomposition import PCA
import os
from tqdm import tqdm
from utils import LABEL_MAPPING, ids2labels, load_best_model

# Konfiguracja
DATA_DIR = "./data/normalized"
MODEL_PATH = (
    "results/FinalSMA/best_model_6.pth"  # Nazwa pliku z best_model_path
)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64  # Batch size z eksperymentu

# Parametry modelu z eksperymentu
NUM_CLASSES = 6
HIDDEN_SIZE = 128
TEXT_COLUMNS = [
    "subject",
    "speaker",
    "job_title",
    "state",
    "party_affiliation",
    "context",
]
NUM_METADATA_COLS = [
    "barely_true_counts",
    "false_counts",
    "half_true_counts",
    "mostly_true_counts",
    "pants_on_fire_counts",
    "grammar_errors",
    "ratio_of_capital_letters",
]
ONE_HOT_COLS = [
    "sentiment",
    "question",
    "curse",
    "emotion",
    "gibberish",
    "offensiveness",
    "political_bias",
]

# One-hot labels z eksperymentu
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

# Oblicz one_hot_metadata_size z eksperymentu
ONE_HOT_METADATA_SIZE = sum([len(x) for x in one_hot_labels.values()])


class LiarPlusSingleRobertaDataset(Dataset):
    def __init__(
        self,
        filepath: str,
        tokenizer,
        str_metadata_cols: list[str],
        num_metadata_cols: list[str],
        one_hot_metadata_cols: list[str],
        max_length: int = 512,
    ):
        # Obsługa CSV
        self.df = pd.read_csv(filepath)

        self.str_metadata_cols = str_metadata_cols
        self.num_metadata_cols = num_metadata_cols
        self.one_hot_metadata_cols = one_hot_metadata_cols

        for column in self.str_metadata_cols:
            if column in self.df.columns:
                self.df[column] = self.df[column].astype(str)

        self.df["statement"] = self.df["statement"].astype(str)
        if "articles" in self.df.columns:
            self.df["articles"] = self.df["articles"].astype(str)
        else:
            self.df["articles"] = ""  # Pusty string jeśli brak kolumny

        self.statement_max_len = max_length // 4
        self.article_max_len = max_length // 4
        self.str_metadata_max_len = max(
            (max_length - self.statement_max_len - self.article_max_len)
            // max(len(str_metadata_cols), 1),
            15,
        )

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df.index)

    def limit_tokens(self, text, max_length=512):
        return self.tokenizer.convert_tokens_to_string(
            self.tokenizer.tokenize(str(text))[:max_length]
        )

    def __getitem__(self, index: int):
        item = self.df.iloc[index]

        input_text = self.limit_tokens(
            f"[STATEMENT] {item['statement']}", self.statement_max_len
        )

        if "articles" in self.df.columns and pd.notna(
            item.get("articles", "")
        ):
            input_text += self.limit_tokens(
                f" [ARTICLE] {item['articles']}",
                self.article_max_len,
            )

        for column in self.str_metadata_cols:
            if column in self.df.columns and pd.notna(item.get(column, "")):
                input_text += self.limit_tokens(
                    f" [{column.upper()}] {item[column]}",
                    self.str_metadata_max_len,
                )

        encoded = self.tokenizer(
            input_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Bezpieczne pobieranie etykiety
        label_key = "label" if "label" in item else "verdict"
        label_value = item[label_key]
        label = LABEL_MAPPING.get(label_value, 0)

        # Dummy values dla metadanych (nie są używane do embeddingów, ale potrzebne dla datasetu)
        num_metadata = [item.get(col, 0.0) for col in NUM_METADATA_COLS]

        one_hot_metadata = []
        for column in ONE_HOT_COLS:
            if column in item and pd.notna(item[column]):
                value = item[column]
                possible_values = len(one_hot_labels[column])
                id_tensor = torch.tensor(label_to_index[column].get(value, 0))
                one_hot_metadata.append(F.one_hot(id_tensor, possible_values))
            else:
                # Dummy one-hot jeśli brak wartości
                possible_values = len(one_hot_labels[column])
                one_hot_metadata.append(torch.zeros(possible_values))

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "num_metadata": torch.tensor(num_metadata).float(),
            "one_hot_metadata": torch.cat(one_hot_metadata, dim=0).float(),
            "label": torch.tensor(label),
            "statement": item["statement"],  # Dodajemy oryginalny tekst
            "split": item.get(
                "split", "unknown"
            ),  # Dodajemy informację o zbiorze
            "label_name": label_value,  # Dodajemy nazwę etykiety
        }


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
        return logits, cls_embedding  # Zwracamy też cls_embedding

    def get_encoder_embeddings(self, input_ids, attention_mask):
        """Funkcja do pobierania tylko embeddingów z encodera"""
        with torch.no_grad():
            outputs = self.encoder(
                input_ids=input_ids, attention_mask=attention_mask
            )
            return outputs.pooler_output  # CLS token embedding

    def roberta_trainable_state(self):
        return {
            name: param
            for name, param in self.encoder.named_parameters()
            if param.requires_grad
        }

    def load_roberta_trainable_state(self, state_dict):
        self.encoder.load_state_dict(state_dict, strict=False)

    def state_for_save(self):
        return {
            "hl_state_dict": self.hl.state_dict(),
            "fc_state_dict": self.fc.state_dict(),
            "roberta_trainable": self.roberta_trainable_state(),
        }

    def load_state_from_save(self, state):
        self.hl.load_state_dict(state["hl_state_dict"])
        self.fc.load_state_dict(state["fc_state_dict"])
        if "roberta_trainable" in state:
            self.load_roberta_trainable_state(state["roberta_trainable"])


def load_datasets(
    data_dir,
    tokenizer,
    str_metadata_cols=None,
    num_metadata_cols=None,
    one_hot_metadata_cols=None,
):
    """Wczytuje wszystkie datasety"""
    files = ["train2.csv", "test2.csv", "val2.csv"]
    datasets = []

    # Domyślne wartości jeśli nie podano
    if str_metadata_cols is None:
        str_metadata_cols = []
    if num_metadata_cols is None:
        num_metadata_cols = []
    if one_hot_metadata_cols is None:
        one_hot_metadata_cols = []

    for file in files:
        file_path = os.path.join(data_dir, file)
        if os.path.exists(file_path):
            # Dodaj informację o zbiorze do dataframe
            temp_df = pd.read_csv(file_path)  # CSV zamiast TSV
            temp_df["split"] = file.replace("2.csv", "")  # .csv zamiast .tsv

            # Zapisz zmodyfikowany dataframe
            temp_path = file_path + "_temp"
            temp_df.to_csv(temp_path, index=False)  # CSV bez separatora tab

            dataset = LiarPlusSingleRobertaDataset(
                temp_path,
                tokenizer,
                str_metadata_cols,
                num_metadata_cols,
                one_hot_metadata_cols,
            )
            datasets.append(dataset)
            print(f"Wczytano {file}: {len(dataset)} rekordów")

            # Usuń tymczasowy plik
            os.remove(temp_path)
        else:
            print(f"Ostrzeżenie: Nie znaleziono pliku {file_path}")

    # Połącz wszystkie datasety
    if datasets:
        combined_data = []
        for dataset in datasets:
            for i in range(len(dataset)):
                combined_data.append(dataset[i])
        return combined_data
    else:
        raise FileNotFoundError(
            "Nie znaleziono żadnych plików CSV w podanym katalogu"
        )


def extract_embeddings_with_custom_model(model, data, device, batch_size=32):
    """Wyciąga embeddingi z custom modelu"""
    model.eval()
    embeddings = []
    labels = []
    statements = []
    splits = []

    # Przygotuj DataLoader
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for batch in tqdm(
            dataloader, desc="Generowanie embeddingów z custom modelu"
        ):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Pobierz embeddingi tylko z encodera
            batch_embeddings = model.get_encoder_embeddings(
                input_ids, attention_mask
            )

            embeddings.append(batch_embeddings.cpu().numpy())
            labels.extend(
                [batch["label_name"][i] for i in range(len(batch["label"]))]
            )
            statements.extend(batch["statement"])
            splits.extend(batch["split"])

    return np.vstack(embeddings), labels, statements, splits


def create_pca_visualization(
    embeddings, labels, split_info, statements, n_components=2
):
    """Tworzy wizualizację PCA embeddingów z ulepszoną paletą kolorów i uporządkowanymi etykietami"""
    pca = PCA(n_components=n_components)
    embeddings_pca = pca.fit_transform(embeddings)

    print(
        f"Wyjaśniona wariancja przez {n_components} składowe PCA: {pca.explained_variance_ratio_.sum():.3f}"
    )

    # Przygotowanie danych do wizualizacji - uporządkowane logicznie
    # Definicja poprawnej kolejności etykiet od najbardziej do najmniej prawdziwych
    label_order = [
        "true",
        "mostly-true",
        "half-true",
        "barely-true",
        "false",
        "pants-fire",
    ]

    # Znajdź etykiety obecne w danych i uporządkuj je zgodnie z label_order
    present_labels = set(labels)
    unique_labels = [label for label in label_order if label in present_labels]

    # Dodaj ewentualne dodatkowe etykiety, które nie są w standardowej kolejności
    additional_labels = sorted(
        [label for label in present_labels if label not in label_order]
    )
    unique_labels.extend(additional_labels)

    print(f"Kolejność etykiet w wizualizacji: {unique_labels}")

    # Definicja kontrastowych kolorów dla etykiet prawdziwości (w odpowiedniej kolejności)
    label_colors = {
        "true": "#2E8B57",  # Sea Green - dla prawdy (najciemniejszy zielony)
        # Lime Green - dla mostly true (jaśniejszy zielony)
        "mostly-true": "#32CD32",
        # Gold - dla half true (żółty/złoty - neutralny)
        "half-true": "#FFD700",
        # Dark Orange - dla barely true (pomarańczowy)
        "barely-true": "#FF8C00",
        "false": "#DC143C",  # Crimson - dla fałszu (czerwony)
        # Dark Red - dla pants on fire (najciemniejszy czerwony)
        "pants-fire": "#8B0000",
    }

    # Fallback colors jeśli etykiety są inne
    fallback_colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    # Przypisz kolory do etykiet
    colors = []
    for i, label in enumerate(unique_labels):
        if label in label_colors:
            colors.append(label_colors[label])
        else:
            colors.append(fallback_colors[i % len(fallback_colors)])

    # Tworzenie wykresu
    plt.figure(figsize=(20, 15))

    # Subplot 1: Kolorowanie według etykiet prawdziwości
    plt.subplot(2, 2, 1)
    for i, label in enumerate(unique_labels):
        mask = np.array(labels) == label
        plt.scatter(
            embeddings_pca[mask, 0],
            embeddings_pca[mask, 1],
            c=colors[i],
            label=label,
            alpha=0.7,
            s=30,
            edgecolors="black",
            linewidth=0.5,
        )

    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.3f})", fontsize=12)
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.3f})", fontsize=12)
    plt.title(
        "Custom Model Embeddings - PCA (według etykiet)",
        fontsize=14,
        fontweight="bold",
    )
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)
    plt.grid(True, alpha=0.3)

    # Subplot 2: Kolorowanie według zbioru z kontrastowymi kolorami
    plt.subplot(2, 2, 2)
    unique_splits = sorted(set(split_info))

    # Kontrastowe kolory dla zbiorów
    split_color_map = {
        "train": "#FF6B6B",  # Czerwony
        "test": "#4ECDC4",  # Turkusowy
        "val": "#45B7D1",  # Niebieski
        "validation": "#45B7D1",  # Alternatywna nazwa dla val
    }

    split_colors = []
    for split in unique_splits:
        if split in split_color_map:
            split_colors.append(split_color_map[split])
        else:
            # Fallback dla nieoczekiwanych nazw
            split_colors.append(
                fallback_colors[len(split_colors) % len(fallback_colors)]
            )

    for i, split in enumerate(unique_splits):
        mask = np.array(split_info) == split
        plt.scatter(
            embeddings_pca[mask, 0],
            embeddings_pca[mask, 1],
            c=split_colors[i],
            label=split,
            alpha=0.7,
            s=30,
            edgecolors="black",
            linewidth=0.5,
        )

    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.3f})", fontsize=12)
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.3f})", fontsize=12)
    plt.title(
        "Custom Model Embeddings - PCA (według zbioru)",
        fontsize=14,
        fontweight="bold",
    )
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    # Subplot 3: Histogram pierwszej składowej z kolorami według etykiet
    plt.subplot(2, 2, 3)
    for i, label in enumerate(unique_labels):
        mask = np.array(labels) == label
        plt.hist(
            embeddings_pca[mask, 0],
            bins=30,
            alpha=0.6,
            label=label,
            color=colors[i],
            edgecolor="black",
            linewidth=0.5,
        )

    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.3f})", fontsize=12)
    plt.ylabel("Częstotliwość", fontsize=12)
    plt.title(
        "Rozkład pierwszej składowej PCA", fontsize=14, fontweight="bold"
    )
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)

    # Subplot 4: Histogram drugiej składowej z kolorami według etykiet
    plt.subplot(2, 2, 4)
    for i, label in enumerate(unique_labels):
        mask = np.array(labels) == label
        plt.hist(
            embeddings_pca[mask, 1],
            bins=30,
            alpha=0.6,
            label=label,
            color=colors[i],
            edgecolor="black",
            linewidth=0.5,
        )

    plt.xlabel(f"PC2 ({pca.explained_variance_ratio_[1]:.3f})", fontsize=12)
    plt.ylabel("Częstotliwość", fontsize=12)
    plt.title("Rozkład drugiej składowej PCA", fontsize=14, fontweight="bold")
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Dodatkowy wykres: większy scatter plot z lepszą czytelności
    plt.figure(figsize=(14, 10))
    for i, label in enumerate(unique_labels):
        mask = np.array(labels) == label
        plt.scatter(
            embeddings_pca[mask, 0],
            embeddings_pca[mask, 1],
            c=colors[i],
            label=label,
            alpha=0.8,
            s=50,
            edgecolors="white",
            linewidth=1,
        )

    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.3f})", fontsize=14)
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.3f})", fontsize=14)
    plt.title(
        "Custom Model Embeddings - PCA (powiększony widok)",
        fontsize=16,
        fontweight="bold",
    )
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=12)
    plt.grid(True, alpha=0.3)

    # Dodaj informacje o liczbie punktów dla każdej etykiety (uporządkowane według unique_labels)
    label_counts = pd.Series(labels).value_counts()
    # Uporządkuj według unique_labels zamiast alfabetycznie
    ordered_counts = [
        (label, label_counts.get(label, 0)) for label in unique_labels
    ]
    info_text = "\n".join(
        [f"{label}: {count}" for label, count in ordered_counts]
    )
    plt.text(
        0.02,
        0.98,
        f"Liczba punktów:\n{info_text}",
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    plt.tight_layout()
    plt.show()

    # Dodatkowy wykres: porównanie zbiorów train/test/val z uporządkowanymi etykietami
    plt.figure(figsize=(16, 6))

    # Subplot dla każdego zbioru
    for idx, split in enumerate(unique_splits):
        plt.subplot(1, len(unique_splits), idx + 1)

        # Filtruj punkty tylko dla danego zbioru
        split_mask = np.array(split_info) == split
        split_embeddings = embeddings_pca[split_mask]
        split_labels = np.array(labels)[split_mask]

        # Narysuj punkty z kolorami według etykiet (w uporządkowanej kolejności)
        for i, label in enumerate(unique_labels):
            label_mask = split_labels == label
            if np.any(label_mask):
                plt.scatter(
                    split_embeddings[label_mask, 0],
                    split_embeddings[label_mask, 1],
                    c=colors[i],
                    label=label,
                    alpha=0.7,
                    s=40,
                    edgecolors="black",
                    linewidth=0.5,
                )

        plt.xlabel(
            f"PC1 ({pca.explained_variance_ratio_[0]:.3f})", fontsize=12
        )
        plt.ylabel(
            f"PC2 ({pca.explained_variance_ratio_[1]:.3f})", fontsize=12
        )
        plt.title(
            f"{split.upper()} set ({np.sum(split_mask)} punktów)",
            fontsize=14,
            fontweight="bold",
        )
        plt.grid(True, alpha=0.3)

        if (
            idx == len(unique_splits) - 1
        ):  # Legenda tylko dla ostatniego subplot
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)

    plt.tight_layout()
    plt.show()

    return pca, embeddings_pca


def load_model(model_path, device):
    """Ładuje zapisany model z parametrami z eksperymentu"""
    # Inicjalizuj tokenizer i encoder
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    encoder = RobertaModel.from_pretrained("roberta-base")

    # Ustawienie parametrów trenowania - trenuje 2 ostatnie warstwy
    for name, param in encoder.named_parameters():
        if name.startswith("encoder.layer.11") or name.startswith("pooler"):
            param.requires_grad = True
        else:
            param.requires_grad = False

    # Inicjalizuj model z parametrami z eksperymentu
    model = LiarPlusSingleFinetunedRoBERTasClassifier(
        encoder,
        len(NUM_METADATA_COLS),  # num_metadata_len
        ONE_HOT_METADATA_SIZE,  # one_hot_metadata_size z eksperymentu
        HIDDEN_SIZE,  # num_hidden z eksperymentu
        NUM_CLASSES,  # num_classes z eksperymentu
    ).to(device)

    # Załaduj wagi używając funkcji load_best_model
    if os.path.exists(model_path):
        load_best_model(model, model_path)
        print(f"Załadowano model z: {model_path}")
    else:
        print(f"Ostrzeżenie: Nie znaleziono modelu w {model_path}")
        print("Używam bazowego RoBERTa bez fine-tuningu")

    return model, tokenizer


def main():
    print("=== Custom Model PCA Visualization ===")
    print(f"Używane urządzenie: {DEVICE}")

    # 1. Załaduj model
    print("\n1. Ładowanie custom modelu...")
    model, tokenizer = load_model(MODEL_PATH, DEVICE)

    # 2. Wczytaj dane
    print("\n2. Wczytywanie danych...")
    data = load_datasets(DATA_DIR, tokenizer)

    print(f"Łącznie wczytano: {len(data)} rekordów")

    # 3. Wyciągnij embeddingi
    print("\n3. Generowanie embeddingów...")
    embeddings, labels, statements, splits = (
        extract_embeddings_with_custom_model(model, data, DEVICE, BATCH_SIZE)
    )

    print(f"Wygenerowano embeddingi o kształcie: {embeddings.shape}")
    print(f"Rozkład etykiet: {pd.Series(labels).value_counts()}")

    # 4. Wizualizacja PCA
    print("\n4. Tworzenie wizualizacji PCA...")
    pca, embeddings_pca = create_pca_visualization(
        embeddings, labels, splits, statements
    )

    # 5. Statystyki
    print("\n=== Statystyki ===")
    print(f"Liczba unikalnych etykiet: {len(set(labels))}")
    print(f"Etykiety: {set(labels)}")
    print(f"Wymiar oryginalnych embeddingów: {embeddings.shape[1]}")
    print(f"Wymiar po PCA: {embeddings_pca.shape[1]}")
    print(f"Wyjaśniona wariancja: {pca.explained_variance_ratio_.sum():.3f}")

    return embeddings, embeddings_pca, labels, splits, pca


if __name__ == "__main__":
    try:
        embeddings, embeddings_pca, labels, splits, pca = main()
        print("\nProgram zakończył się pomyślnie!")
    except Exception as e:
        print(f"Błąd: {e}")
        import traceback

        traceback.print_exc()
