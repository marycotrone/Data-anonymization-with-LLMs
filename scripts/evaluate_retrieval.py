"""
Paraphrase Retrieval Evaluation Script

This script evaluates the privacy of anonymization techniques by measuring
how easy it is to retrieve the original sentence from a paraphrased version.

Lower accuracy = better privacy (harder to find the original).

Usage:
    python scripts/evaluate_retrieval.py \\
        --original data/dataset_train_original.csv \\
        --paraphrased results/dataset_train_llm_04_09.csv \\
        --config configs/config.yaml
"""

import argparse
import pandas as pd
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from metrics import AnonymizationMetrics
from utils import load_config


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description='Evaluate paraphrase retrieval for privacy assessment'
    )
    parser.add_argument(
        '--original',
        type=str,
        required=True,
        help='Path to original dataset CSV'
    )
    parser.add_argument(
        '--paraphrased',
        type=str,
        required=True,
        help='Path to paraphrased/anonymized dataset CSV'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--text-column',
        type=str,
        default='text',
        help='Name of the text column in CSV'
    )
    parser.add_argument(
        '--k-values',
        type=int,
        nargs='+',
        default=[1, 5, 10],
        help='K values for Accuracy@k metrics'
    )
    parser.add_argument(
        '--sample-size',
        type=int,
        default=None,
        help='Number of samples to evaluate (None = all)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    print("Loading configuration...")
    config = load_config(args.config)
    
    # Load datasets
    print(f"\nLoading original dataset from: {args.original}")
    df_original = pd.read_csv(args.original)
    
    print(f"Loading paraphrased dataset from: {args.paraphrased}")
    df_paraphrased = pd.read_csv(args.paraphrased)
    
    # Check text column exists
    if args.text_column not in df_original.columns:
        raise ValueError(f"Column '{args.text_column}' not found in original dataset")
    if args.text_column not in df_paraphrased.columns:
        raise ValueError(f"Column '{args.text_column}' not found in paraphrased dataset")
    
    # Extract text
    original_sentences = df_original[args.text_column].tolist()
    paraphrased_sentences = df_paraphrased[args.text_column].tolist()
    
    # Apply sample size if specified
    if args.sample_size is not None:
        print(f"\nUsing sample size: {args.sample_size}")
        original_sentences = original_sentences[:args.sample_size]
        paraphrased_sentences = paraphrased_sentences[:args.sample_size]
    
    print(f"\nDataset sizes:")
    print(f"  Original: {len(original_sentences)} sentences")
    print(f"  Paraphrased: {len(paraphrased_sentences)} sentences")
    
    if len(original_sentences) != len(paraphrased_sentences):
        print("\n⚠️  Warning: Dataset sizes don't match!")
        print("   The evaluation assumes paraphrased[i] corresponds to original[i]")
        min_size = min(len(original_sentences), len(paraphrased_sentences))
        print(f"   Using first {min_size} samples from both datasets")
        original_sentences = original_sentences[:min_size]
        paraphrased_sentences = paraphrased_sentences[:min_size]
    
    # Initialize metrics module
    print("\n" + "="*60)
    print("INITIALIZING EVALUATION")
    print("="*60)
    
    metrics = AnonymizationMetrics(
        sbert_model=config['metrics']['sbert_model'],
        spacy_model=config['metrics']['spacy_model'],
        verbose=True
    )
    
    # Evaluate retrieval
    print("\n" + "="*60)
    print("STARTING RETRIEVAL EVALUATION")
    print("="*60)
    
    results = metrics.evaluate_paraphrase_retrieval(
        original_sentences=original_sentences,
        paraphrased_sentences=paraphrased_sentences,
        k_values=args.k_values,
        show_progress=True
    )
    
    # Print final summary
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"\nDataset: {args.paraphrased}")
    print(f"Samples evaluated: {len(paraphrased_sentences)}")
    print("\nRetrieval Accuracy (Lower = Better Privacy):")
    for k in args.k_values:
        accuracy = results[f'Accuracy@{k}']
        print(f"  Accuracy@{k:2d}: {accuracy:6.2f}%")
    
    # Interpretation
    print("\nInterpretation:")
    acc1 = results['Accuracy@1']
    if acc1 < 10:
        privacy_level = "EXCELLENT"
    elif acc1 < 30:
        privacy_level = "GOOD"
    elif acc1 < 50:
        privacy_level = "MODERATE"
    elif acc1 < 70:
        privacy_level = "WEAK"
    else:
        privacy_level = "POOR"
    
    print(f"  Privacy Level: {privacy_level}")
    print(f"  Only {acc1:.1f}% of paraphrases can be exactly matched to originals")
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
