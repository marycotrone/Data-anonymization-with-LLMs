# Data Anonymization with LLMs

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Ollama Setup (per LLM anonymization)

```bash
bash scripts/setup_ollama.sh                # default: llama3.2
bash scripts/setup_ollama.sh gemma2:2b      # oppure un altro modello
bash scripts/setup_ollama.sh mistral
```

## Baseline (EDA / KNEO)

```bash
python scripts/run_baseline.py --method both --dataset validation
python scripts/run_baseline.py --method eda --dataset train
python scripts/run_baseline.py --method kneo --dataset test --sample 100
```

## LLM Anonymization

```bash
python scripts/run_llm.py --model gemma2:2b --dataset validation
python scripts/run_llm.py --model llama3.2 --dataset train --sample 50
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

## Notebook Demo

```bash
jupyter notebook notebooks/demo.ipynb
```