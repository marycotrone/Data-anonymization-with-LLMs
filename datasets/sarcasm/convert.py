import pandas as pd

# ── File 1: task_A_En_test.csv ──────────────────────────────────────────────
test = pd.read_csv("task_A_En_test.csv", usecols=["text", "sarcastic"])

test = test.rename(columns={"sarcastic": "label", "text": "text"})
test["label"] = test["label"].map({1: "SARCASM", 0: "NOT_SARCASM"})
test = test[["label", "text"]]

test = test.sample(frac=1, random_state=42).reset_index(drop=True)
test.to_csv("dataset_test_original.csv", index=False)
print(f"Test:  {len(test)} rows -> dataset_test_original.csv")


# ── File 2: train.En.csv ─────────────────────────────────────────────────────
train = pd.read_csv("train.En.csv", usecols=["tweet", "sarcastic"])

train = train.rename(columns={"sarcastic": "label", "tweet": "text"})
train["label"] = train["label"].map({1: "SARCASM", 0: "NOT_SARCASM"})
train = train[["label", "text"]]

train = train.sample(frac=1, random_state=42).reset_index(drop=True)
train.to_csv("dataset_train_original.csv", index=False)
print(f"Train: {len(train)} rows -> dataset_train_original.csv")

