# Data Anonymization with LLMs

## Requisiti

- **Python 3.11** (gensim non è compatibile con Python 3.12+)
- **Ollama** (solo per LLM anonymization)

---

## Installazione

### 1. Python 3.11

```bash
brew install python@3.11
```

### 2. Virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 3. Dipendenze Python

```bash
pip install -r requirements.txt
```

### 4. Modello spaCy

```bash
python -m spacy download en_core_web_sm
```

### 5. Risorse NLTK (una volta sola)

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
```

---

## Dati

Posizionare i file CSV nella cartella `data/`:

```
data/
  dataset_train_original.csv
  dataset_validation_original.csv
  dataset_test_original.csv
```

---

## Ollama Setup (per LLM anonymization)

```bash
brew install ollama
ollama serve                # avviare in un terminale separato e lasciare aperto
ollama pull gemma2:2b
ollama pull llama3.1:8b
```

---

## Notebook Demo

Selezionare il kernel `.venv` in VSCode, poi eseguire le celle in ordine.

Oppure da terminale:

```bash
jupyter notebook notebooks/demo.ipynb
```

> **Nota:** KNEO scarica GloVe (~800MB) al primo run.

---

## Baseline (EDA / KNEO)

```bash
python scripts/run_baseline.py --method both --dataset validation
python scripts/run_baseline.py --method eda --dataset train
python scripts/run_baseline.py --method kneo --dataset test --sample 100
```

## LLM Anonymization

```bash
python scripts/run_llm.py --model gemma2:2b --dataset validation
python scripts/run_llm.py --model llama3.1:8b --dataset train --sample 50
python scripts/run_llm.py --model mistral --prompt-style paraphrase
```

## Retrieval Evaluation

```bash
python scripts/evaluate_retrieval.py \
    --original data/dataset_train_original.csv \
    --paraphrased results/dataset_train_llm_04_09.csv
```

### Batch Retrieval Evaluation

```bash
python scripts/batch_evaluate_retrieval.py \
    --original data/dataset_train_original.csv \
    --paraphrased-dir results/ \
    --pattern "dataset_train_*_*.csv"
```

## Train Sentiment Classifier

```bash
python scripts/train_sentiment_classifier.py \
    --train-data data/dataset_train_original.csv \
    --val-data data/dataset_validation_original.csv \
    --test-data data/dataset_test_original.csv \
    --output best_model_sentiment.pt \
    --results-dir results
```
