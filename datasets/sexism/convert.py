import pandas as pd

# Load relevant columns (include "split" to partition rows)
df = pd.read_csv("edos_labelled_aggregated.csv", usecols=["text", "label_sexist", "split"])

# Rename and remap labels
df = df.rename(columns={"label_sexist": "label"})
df["label"] = df["label"].map({"sexist": "SEXIST", "not sexist": "NOT_SEXIST"})
df = df[["label", "text", "split"]]

# Split according to the "split" column
mapping = {
    "train": "dataset_train_original.csv",
    "dev":   "dataset_validation_original.csv",
    "test":  "dataset_test_original.csv",
}

for split_name, filename in mapping.items():
    subset = df[df["split"] == split_name][["label", "text"]]
    subset.sample(frac=1, random_state=42).reset_index(drop=True).to_csv(filename, index=False)
    print(f"{split_name}: {len(subset)} rows -> {filename}")