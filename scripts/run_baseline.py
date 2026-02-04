#!/usr/bin/env python3
"""
Main script to run anonymization with baseline techniques (EDA, KNEO)
and evaluate the results.

Usage:
    python scripts/run_baseline.py --config configs/config.yaml
    python scripts/run_baseline.py --method eda --dataset train
    python scripts/run_baseline.py --method kneo --sample 100
"""

import argparse
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eda_anonymizer import EDAAnonymizer
from kneo_anonymizer import KNEOAnonymizer
from metrics import AnonymizationMetrics
from utils import (
    load_config, 
    load_all_datasets,
    create_output_dir,
    set_all_seeds,
    save_anonymized_dataset,
    save_metrics,
    print_comparison_table
)


def main():
    parser = argparse.ArgumentParser(description="Run baseline anonymization")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["eda", "kneo", "both"],
        default="both",
        help="Method to use"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["train", "validation", "test"],
        default="validation",
        help="Dataset to process"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Number of samples to process (for quick tests)"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    print("📋 Loading configuration...")
    config = load_config(args.config)
    
    # Set seeds for reproducibility
    set_all_seeds(config["general"]["seed"])
    
    # Load dataset
    print(f"\n Loading dataset: {args.dataset}...")
    sample_sizes = {args.dataset: args.sample} if args.sample else None
    datasets = load_all_datasets(config, sample_sizes=sample_sizes)
    
    if args.dataset not in datasets:
        print(f" Dataset {args.dataset} not found!")
        sys.exit(1)
    
    sentences, labels = datasets[args.dataset]
    print(f"   Loaded {len(sentences)} samples")
    
    # Create output directory
    output_dir = create_output_dir(config)
    
    # Initialize metrics
    print("\n Initializing metrics...")
    metrics = AnonymizationMetrics(
        sbert_model=config["metrics"]["sbert_model"],
        spacy_model=config["metrics"]["spacy_model"],
        verbose=config["general"]["verbose"]
    )
    
    results = {}
    
    # === EDA ===
    if args.method in ["eda", "both"]:
        print("\n" + "="*60)
        print("EASY DATA AUGMENTATION (EDA)")
        print("="*60)
        
        eda = EDAAnonymizer(seed=config["eda"]["seed"])
        
        anonymized_eda = eda.anonymize_batch(
            sentences,
            alpha_sr=config["eda"]["alpha_sr"],
            alpha_ri=config["eda"]["alpha_ri"],
            alpha_rs=config["eda"]["alpha_rs"],
            alpha_rd=config["eda"]["alpha_rd"],
            show_progress=config["general"]["show_progress"]
        )
        
        # Evaluate
        print("\nEvaluating EDA...")
        eda_metrics = metrics.evaluate_all(sentences, anonymized_eda)
        results["EDA"] = eda_metrics
        
        # Save
        if config["output"]["save_anonymized"]:
            save_anonymized_dataset(anonymized_eda, labels, output_dir, f'eda_{args.dataset}')
        
        # Show examples
        n_examples = config['general'].get('n_examples', 5)
        print(f"\nExamples ({n_examples} samples):")
        for i in range(min(n_examples, len(sentences))):
            print(f"\n{i+1}. Original: {sentences[i]}")
            print(f"   EDA:       {anonymized_eda[i]}")
    
    # === KNEO ===
    if args.method in ["kneo", "both"]:
        print("\n" + "="*60)
        print("KNOWLEDGE-BASED NEIGHBOR OPERATION (KNEO)")
        print("="*60)
        
        kneo = KNEOAnonymizer(
            embedding_model=config["kneo"]["embedding_model"],
            seed=config["kneo"]["seed"],
            verbose=config["general"]["verbose"]
        )
        
        anonymized_kneo = kneo.anonymize_batch(
            sentences,
            k=config["kneo"]["k"],
            strategy=config["kneo"]["strategy"],
            show_progress=config["general"]["show_progress"]
        )
        
        # Evaluate
        print("\nEvaluating KNEO...")
        kneo_metrics = metrics.evaluate_all(sentences, anonymized_kneo)
        results["KNEO"] = kneo_metrics
        
        # Cache statistics
        cache_stats = kneo.get_cache_stats()
        print(f"\nCache Stats: {cache_stats}")
        
        # Save
        if config["output"]["save_anonymized"]:
            save_anonymized_dataset(anonymized_kneo, labels, output_dir, f'kneo_{args.dataset}')
        
        # Show examples
        n_examples = config['general'].get('n_examples', 5)
        print(f"\nExamples ({n_examples} samples):")
        for i in range(min(n_examples, len(sentences))):
            print(f"\n{i+1}. Original: {sentences[i]}")
            print(f"   KNEO:      {anonymized_kneo[i]}")
    
    # === SUMMARY ===
    if results:
        print("\n" + "="*60)
        print("METRICS SUMMARY")
        print("="*60)
        
        print_comparison_table(results)
        
        # Save metrics
        if config["output"]["save_metrics"]:
            save_metrics(results, output_dir, f'baseline_{args.dataset}')
    
    print("\nCompleted!")


if __name__ == "__main__":
    main()
