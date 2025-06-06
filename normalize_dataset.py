from sklearn.preprocessing import StandardScaler
import pandas as pd


if __name__ == "__main__":
    train_df = pd.read_csv("data/train2.tsv", sep="\t")
    test_df = pd.read_csv("data/test2.tsv", sep="\t")
    val_df = pd.read_csv("data/val2.tsv", sep="\t")

    cols_to_norm = [
        "barely_true_counts",
        "false_counts",
        "half_true_counts",
        "mostly_true_counts",
        "pants_on_fire_counts",
        "grammar_errors",
        "ratio_of_capital_letters",
    ]

    scaler = StandardScaler()

    train_df[cols_to_norm] = scaler.fit_transform(train_df[cols_to_norm])
    test_df[cols_to_norm] = scaler.transform(test_df[cols_to_norm])
    val_df[cols_to_norm] = scaler.transform(val_df[cols_to_norm])

    # Naprawa danych
    train_df.loc[train_df["curse"] == "Curse", "curse"] = "curse"
    train_df["curse"] = train_df["curse"].fillna("non-curse")
    test_df.loc[test_df["curse"] == "Curse", "curse"] = "curse"
    test_df["curse"] = test_df["curse"].fillna("non-curse")
    val_df.loc[val_df["curse"] == "Curse", "curse"] = "curse"
    val_df["curse"] = val_df["curse"].fillna("non-curse")

    train_df.to_csv("data/normalized/train2.csv", index=None)
    test_df.to_csv("data/normalized/test2.csv", index=None)
    val_df.to_csv("data/normalized/val2.csv", index=None)
