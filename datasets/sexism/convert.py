import pandas as pd
from sklearn.model_selection import train_test_split

# Load and select relevant columns
df = pd.read_csv("edos_labelled_aggregated.csv", usecols=["text", "label_sexist"])

# Rename and remap labels
df = df.rename(columns={"label_sexist": "label"})
df["label"] = df["label"].map({"sexist": "SEXIST", "not sexist": "NOT_SEXIST"})

# Shuffle
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Split into train (33%), val (33%), test (33%)
train, temp = train_test_split(df, test_size=0.66, random_state=42)
val, test = train_test_split(temp, test_size=0.50, random_state=42)

# Save
train.to_csv("dataset_train_original.csv", index=False)
val.to_csv("dataset_validation_original.csv", index=False)
test.to_csv("dataset_test_original.csv", index=False)

print(f"Train: {len(train)} rows")
print(f"Val:   {len(val)} rows")
print(f"Test:  {len(test)} rows")
