"""
Classification Task Evaluation
Trains a sentiment classifier on each anonymized dataset to evaluate
how different anonymization methods affect downstream task performance.

Model: Twitter-RoBERTa-base (cardiffnlp/twitter-roberta-base-2022-154m)
Method: K-Fold Cross-Validation with Ensemble (Soft Voting)
"""

import csv
import gc
import random
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, accuracy_score
)
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    get_linear_schedule_with_warmup
)
from tqdm import tqdm

warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# Default hyper-parameters
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = dict(
    model_name     = "cardiffnlp/twitter-roberta-base-2022-154m",
    max_len        = 80,
    batch_size     = 32,
    epochs         = 5,
    n_folds        = 3,
    learning_rate  = 2e-5,
    warmup_ratio   = 0.1,
    dropout        = 0.2,
    weight_decay   = 0.1,
    smoothing      = 0.1,
    patience       = 2,
    clip_value     = 1.0,
    seed           = 42,
    results_dir    = "../results",
)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_test_set(data_dir, filename, text_col, label_col):
    """Load test set from CSV file."""
    path = os.path.join(data_dir, filename)
    texts, labels = [], []
    with open(path, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            texts.append(row[text_col])
            labels.append(row[label_col])
    print(f"Test set loaded: {len(texts)} samples from {path}")
    return texts, labels


def prepare_classification_datasets(val_texts, val_labels, all_anonymized, all_orig_labels):
    """Build dict {name: (texts, labels)} for original + all anonymized configs."""
    datasets = {'Original': (val_texts, val_labels)}
    for name, anonymized in all_anonymized.items():
        datasets[name] = (anonymized, all_orig_labels[name])
    print(f"Classification datasets prepared: {list(datasets.keys())}")
    return datasets


def build_label_map(labels):
    """Return label_map dict and num_classes."""
    label_map = {label: idx for idx, label in enumerate(sorted(set(labels)))}
    print(f"Label map: {label_map}  ({len(label_map)} classes)")
    return label_map, len(label_map)


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len, label_map):
        self.texts     = texts
        self.labels    = [label_map[l] for l in labels]
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]),
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids':      enc['input_ids'].flatten(),
            'attention_mask': enc['attention_mask'].flatten(),
            'labels':         torch.tensor(self.labels[idx], dtype=torch.long)
        }


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def _get_model(num_labels, cfg, device):
    model_cfg = AutoConfig.from_pretrained(cfg['model_name'], num_labels=num_labels)
    model_cfg.hidden_dropout_prob             = cfg['dropout']
    model_cfg.attention_probs_dropout_prob    = cfg['dropout']
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg['model_name'], config=model_cfg
    )
    for name, param in model.named_parameters():
        if "encoder.layer" in name or "transformer.layer" in name:
            try:
                if int(name.split(".")[3]) < 4:
                    param.requires_grad = False
            except (IndexError, ValueError):
                pass
    return model.to(device)


def _train_epoch(model, loader, optimizer, scheduler, criterion, device, clip):
    model.train()
    losses, preds, trues = [], [], []
    for d in tqdm(loader, desc="Training", leave=False):
        ids  = d["input_ids"].to(device)
        mask = d["attention_mask"].to(device)
        labs = d["labels"].to(device)

        out  = model(input_ids=ids, attention_mask=mask)
        loss = criterion(out.logits, labs)
        _, pred = torch.max(out.logits, dim=1)

        preds.extend(pred.cpu().numpy())
        trues.extend(labs.cpu().numpy())
        losses.append(loss.item())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return np.mean(losses), f1_score(trues, preds, average='macro')


def _eval_model(model, loader, criterion, device):
    model.eval()
    losses, preds, trues = [], [], []
    with torch.no_grad():
        for d in loader:
            ids  = d["input_ids"].to(device)
            mask = d["attention_mask"].to(device)
            labs = d["labels"].to(device)

            out  = model(input_ids=ids, attention_mask=mask)
            loss = criterion(out.logits, labs)
            losses.append(loss.item())

            _, pred = torch.max(out.logits, dim=1)
            preds.extend(pred.cpu().numpy())
            trues.extend(labs.cpu().numpy())

    return np.mean(losses), accuracy_score(trues, preds), f1_score(trues, preds, average='macro')


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_kfold_ensemble(dataset_name, texts, labels, label_map, num_classes,
                         tokenizer, cfg=None):
    """Train K-fold ensemble for one dataset. Returns list of per-fold best F1."""
    cfg    = {**DEFAULT_CONFIG, **(cfg or {})}
    device = get_device()

    print(f"\n{'='*60}\nTraining on {dataset_name}\n{'='*60}")

    numeric_labels = np.array([label_map[l] for l in labels])
    texts_array    = np.array(texts)

    class_weights   = compute_class_weight('balanced',
                                           classes=np.unique(numeric_labels),
                                           y=numeric_labels)
    weights_tensor  = torch.tensor(class_weights, dtype=torch.float).to(device)
    criterion       = torch.nn.CrossEntropyLoss(weight=weights_tensor,
                                                label_smoothing=cfg['smoothing'])

    skf          = StratifiedKFold(n_splits=cfg['n_folds'], shuffle=True,
                                   random_state=cfg['seed'])
    fold_results = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(texts_array, numeric_labels)):
        print(f"\n--- Fold {fold+1}/{cfg['n_folds']} ---")

        X_tr = texts_array[tr_idx].tolist()
        X_va = texts_array[va_idx].tolist()
        y_tr = [labels[i] for i in tr_idx]
        y_va = [labels[i] for i in va_idx]

        tr_loader = DataLoader(
            TextDataset(X_tr, y_tr, tokenizer, cfg['max_len'], label_map),
            batch_size=cfg['batch_size'], shuffle=True)
        va_loader = DataLoader(
            TextDataset(X_va, y_va, tokenizer, cfg['max_len'], label_map),
            batch_size=cfg['batch_size'], shuffle=False)

        model     = _get_model(num_classes, cfg, device)
        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                          lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
        total_steps = len(tr_loader) * cfg['epochs']
        scheduler   = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(cfg['warmup_ratio'] * total_steps),
            num_training_steps=total_steps
        )

        best_f1 = 0
        patience_counter = 0

        for epoch in range(cfg['epochs']):
            tr_loss, tr_f1 = _train_epoch(model, tr_loader, optimizer, scheduler,
                                           criterion, device, cfg['clip_value'])
            va_loss, va_acc, va_f1 = _eval_model(model, va_loader, criterion, device)
            print(f"Epoch {epoch+1}/{cfg['epochs']} | "
                  f"Train Loss: {tr_loss:.4f}  F1: {tr_f1:.4f} | "
                  f"Val Loss: {va_loss:.4f}  F1: {va_f1:.4f}")

            if va_f1 > best_f1:
                best_f1 = va_f1
                patience_counter = 0
                save_path = os.path.join(cfg['results_dir'],
                                         f"best_model_{dataset_name}_fold_{fold+1}.pt")
                torch.save(model.state_dict(), save_path)
            else:
                patience_counter += 1
                if patience_counter >= cfg['patience']:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        print(f"Best Val F1 Fold {fold+1}: {best_f1:.4f}")
        fold_results.append(best_f1)

        del model, optimizer, scheduler, tr_loader, va_loader
        torch.cuda.empty_cache()
        gc.collect()

    print(f"\n{dataset_name} — Avg F1: {np.mean(fold_results):.4f}")
    return fold_results


def train_all_datasets(classification_datasets, label_map, num_classes, tokenizer, cfg=None):
    """Train K-fold ensemble for every dataset. Returns all_training_results dict."""
    all_training_results = {}
    for name, (texts, labels) in classification_datasets.items():
        fold_scores = train_kfold_ensemble(name, texts, labels, label_map,
                                           num_classes, tokenizer, cfg)
        all_training_results[name] = {
            'fold_scores': fold_scores,
            'avg_f1':      np.mean(fold_scores),
            'std_f1':      np.std(fold_scores)
        }

    print("\n" + "="*60 + "\nTRAINING SUMMARY\n" + "="*60)
    for name, res in all_training_results.items():
        print(f"{name:20s}  Avg F1: {res['avg_f1']:.4f} ± {res['std_f1']:.4f}")
    return all_training_results


# ---------------------------------------------------------------------------
# Ensemble prediction & evaluation
# ---------------------------------------------------------------------------

def ensemble_predict(dataset_name, test_texts, test_labels, label_map, num_classes,
                     tokenizer, cfg=None):
    """Soft-voting ensemble from all K-fold saved models."""
    cfg    = {**DEFAULT_CONFIG, **(cfg or {})}
    device = get_device()

    print(f"\n{'='*60}\nEnsemble Prediction: {dataset_name}\n{'='*60}")

    test_loader = DataLoader(
        TextDataset(test_texts, test_labels, tokenizer, cfg['max_len'], label_map),
        batch_size=cfg['batch_size'], shuffle=False)

    all_logits = []
    for fold in range(cfg['n_folds']):
        model_path = os.path.join(cfg['results_dir'],
                                  f"best_model_{dataset_name}_fold_{fold+1}.pt")
        try:
            model = _get_model(num_classes, cfg, device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()

            fold_logits = []
            with torch.no_grad():
                for d in test_loader:
                    out = model(input_ids=d["input_ids"].to(device),
                                attention_mask=d["attention_mask"].to(device))
                    fold_logits.append(out.logits.cpu())

            all_logits.append(torch.cat(fold_logits, dim=0))
            del model
            torch.cuda.empty_cache()
            print(f"  Fold {fold+1} loaded ✅")
        except FileNotFoundError:
            print(f"  ⚠️  Model not found: {model_path}")

    if not all_logits:
        print(f"No models found for {dataset_name}")
        return None, None

    avg_logits   = torch.mean(torch.stack(all_logits), dim=0)
    predictions  = torch.argmax(avg_logits, dim=1).numpy()
    true_labels  = np.array([label_map[l] for l in test_labels])
    return predictions, true_labels


def evaluate_all_datasets(classification_datasets, test_texts, test_labels,
                           label_map, num_classes, tokenizer, cfg=None):
    """Run ensemble_predict for every dataset and collect metrics."""
    all_test_results = {}
    for name in classification_datasets:
        preds, trues = ensemble_predict(name, test_texts, test_labels,
                                        label_map, num_classes, tokenizer, cfg)
        if preds is not None:
            all_test_results[name] = {
                'predictions': preds,
                'accuracy':    accuracy_score(trues, preds),
                'f1_macro':    f1_score(trues, preds, average='macro'),
                'f1_weighted': f1_score(trues, preds, average='weighted'),
            }
            print(f"\n{name}  Accuracy: {all_test_results[name]['accuracy']:.4f}  "
                  f"F1 Macro: {all_test_results[name]['f1_macro']:.4f}")
    print("\n✅ Test set evaluation completed")
    return all_test_results


# ---------------------------------------------------------------------------
# Results reporting
# ---------------------------------------------------------------------------

def print_results_table(all_test_results):
    """Print sorted results table and performance retention vs Original."""
    rows = [
        {'Dataset': name, 'Accuracy': r['accuracy'],
         'F1 Macro': r['f1_macro'], 'F1 Weighted': r['f1_weighted']}
        for name, r in all_test_results.items()
    ]
    df = pd.DataFrame(rows).sort_values('F1 Macro', ascending=False)
    print("\n" + "="*70 + "\nCLASSIFICATION RESULTS ON TEST SET\n" + "="*70)
    try:
        from IPython.display import display
        display(df.round(4))
    except Exception:
        print(df.round(4).to_string(index=False))

    if 'Original' in all_test_results:
        orig_f1 = all_test_results['Original']['f1_macro']
        print("\nPerformance Retention (vs Original):")
        for name, r in all_test_results.items():
            if name != 'Original':
                print(f"  {name:20s}: {r['f1_macro']/orig_f1*100:.2f}%")
    return df


def _get_bar_color(name):
    """Per-group colors consistent with the qualitative palette."""
    n = name.upper()
    if n == 'ORIGINAL':  return 'hotpink'
    if n.startswith('EDA'):   return '#c0392b'
    if n.startswith('KNEO'):  return '#2980b9'
    if n.startswith('GEMMA'): return '#27ae60'
    if n.startswith('LLAMA'): return '#8e44ad'
    return 'gray'


def plot_classification_comparison(all_test_results, save_path=None):
    """Bar charts: Accuracy and F1 Macro for each dataset."""
    df = pd.DataFrame([
        {'Dataset': n, 'Accuracy': r['accuracy'], 'F1 Macro': r['f1_macro']}
        for n, r in all_test_results.items()
    ]).sort_values('F1 Macro', ascending=False)

    colors   = [_get_bar_color(d) for d in df['Dataset']]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, col, title in zip(axes,
                               ['Accuracy', 'F1 Macro'],
                               ['Classification Accuracy on Test Set',
                                'F1 Macro Score on Test Set']):
        bars = ax.bar(df['Dataset'], df[col], color=colors, alpha=0.8, edgecolor='black')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylabel(col, fontsize=11)
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=0.3)
        ax.set_xticklabels(df['Dataset'], rotation=45, ha='right', fontsize=8)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., h + 0.01,
                    f'{h:.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    out = save_path or "../results/classification_comparison.png"
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f"Saved: {out}")
    plt.show()


def plot_confusion_matrices(all_test_results, test_labels, label_map, save_path=None):
    """One confusion matrix per dataset."""
    reverse_map  = {v: k for k, v in label_map.items()}
    label_names  = [reverse_map[i] for i in sorted(reverse_map)]
    true_numeric = np.array([label_map[l] for l in test_labels])

    n   = len(all_test_results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (name, res) in zip(axes, all_test_results.items()):
        cm = confusion_matrix(true_numeric, res['predictions'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=label_names, yticklabels=label_names,
                    ax=ax, cbar=False)
        ax.set_title(f"{name}\nF1: {res['f1_macro']:.3f}", fontsize=11, fontweight='bold')
        ax.set_xlabel('Predicted', fontsize=10)
        ax.set_ylabel('True', fontsize=10)

    plt.tight_layout()
    out = save_path or "../results/confusion_matrices.png"
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f"Saved: {out}")
    plt.show()


def print_classification_reports(all_test_results, test_labels, label_map):
    """Detailed per-class classification report for each dataset."""
    reverse_map  = {v: k for k, v in label_map.items()}
    label_names  = [reverse_map[i] for i in sorted(reverse_map)]
    true_numeric = np.array([label_map[l] for l in test_labels])

    for name, res in all_test_results.items():
        print("\n" + "="*70)
        print(f"CLASSIFICATION REPORT: {name}")
        print("="*70)
        print(classification_report(true_numeric, res['predictions'],
                                    target_names=label_names, digits=4))
