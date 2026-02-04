"""
Batch Retrieval Evaluation Script

This script evaluates multiple paraphrased datasets at once, comparing
different anonymization configurations.

Usage:
    python scripts/batch_evaluate_retrieval.py \\
        --original data/dataset_train_original.csv \\
        --paraphrased-dir results/ \\
        --pattern "dataset_train_*_*.csv" \\
        --config configs/config.yaml
"""

import argparse
import pandas as pd
import sys
from pathlib import Path
from glob import glob

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from metrics import AnonymizationMetrics
from utils import load_config


def extract_params_from_filename(filename):
    """
    Extract parameters from filename like 'dataset_train_04_09.csv'.
    
    Args:
        filename: Filename string
        
    Returns:
        Parameter string (e.g., '04_09') or filename if pattern not matched
    """
    stem = Path(filename).stem
    parts = stem.split('_')
    
    # Try to find parameter pattern (numbers separated by underscore)
    if len(parts) >= 2:
        # Get last 2 parts if they look like numbers
        try:
            param1 = parts[-2]
            param2 = parts[-1]
            # Check if they're numeric-ish
            if param1.replace('.', '').replace('-', '').isdigit() and \
               param2.replace('.', '').replace('-', '').isdigit():
                return f"{param1}_{param2}"
        except:
            pass
    
    return stem


def main():
    """Main batch evaluation function."""
    parser = argparse.ArgumentParser(
        description='Batch evaluate paraphrase retrieval for multiple datasets'
    )
    parser.add_argument(
        '--original',
        type=str,
        required=True,
        help='Path to original dataset CSV'
    )
    parser.add_argument(
        '--paraphrased-dir',
        type=str,
        required=True,
        help='Directory containing paraphrased datasets'
    )
    parser.add_argument(
        '--pattern',
        type=str,
        default='*.csv',
        help='Glob pattern for paraphrased files (e.g., "dataset_train_*.csv")'
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
    parser.add_argument(
        '--output',
        type=str,
        default='results/retrieval_comparison.csv',
        help='Output CSV file for comparison results'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    print("Loading configuration...")
    config = load_config(args.config)
    
    # Load original dataset
    print(f"\nLoading original dataset from: {args.original}")
    df_original = pd.read_csv(args.original)
    original_sentences = df_original[args.text_column].tolist()
    
    if args.sample_size is not None:
        original_sentences = original_sentences[:args.sample_size]
    
    print(f"Original dataset: {len(original_sentences)} sentences")
    
    # Find paraphrased files
    search_pattern = str(Path(args.paraphrased_dir) / args.pattern)
    paraphrased_files = sorted(glob(search_pattern))
    
    if not paraphrased_files:
        print(f"\n❌ No files found matching pattern: {search_pattern}")
        return
    
    print(f"\nFound {len(paraphrased_files)} paraphrased datasets:")
    for f in paraphrased_files:
        print(f"  - {Path(f).name}")
    
    # Initialize metrics module
    print("\n" + "="*60)
    print("INITIALIZING EVALUATION")
    print("="*60)
    
    metrics = AnonymizationMetrics(
        sbert_model=config['metrics']['sbert_model'],
        spacy_model=config['metrics']['spacy_model'],
        verbose=True
    )
    
    # Store all results
    all_results = []
    
    # Evaluate each paraphrased dataset
    for paraphrased_file in paraphrased_files:
        filename = Path(paraphrased_file).name
        params = extract_params_from_filename(filename)
        
        print("\n" + "="*60)
        print(f"EVALUATING: {filename}")
        print(f"Parameters: {params}")
        print("="*60)
        
        # Load paraphrased dataset
        df_paraphrased = pd.read_csv(paraphrased_file)
        paraphrased_sentences = df_paraphrased[args.text_column].tolist()
        
        if args.sample_size is not None:
            paraphrased_sentences = paraphrased_sentences[:args.sample_size]
        
        # Ensure same size
        min_size = min(len(original_sentences), len(paraphrased_sentences))
        orig_subset = original_sentences[:min_size]
        para_subset = paraphrased_sentences[:min_size]
        
        print(f"Evaluating {min_size} sentence pairs...")
        
        # Evaluate retrieval
        results = metrics.evaluate_paraphrase_retrieval(
            original_sentences=orig_subset,
            paraphrased_sentences=para_subset,
            k_values=args.k_values,
            show_progress=True
        )
        
        # Store results
        result_row = {
            'filename': filename,
            'parameters': params,
            'num_samples': min_size
        }
        result_row.update(results)
        all_results.append(result_row)
    
    # Create comparison DataFrame
    df_comparison = pd.DataFrame(all_results)
    
    # Sort by Accuracy@1 (ascending = better privacy)
    df_comparison = df_comparison.sort_values('Accuracy@1')
    
    # Print comparison table
    print("\n" + "="*60)
    print("FINAL COMPARISON (sorted by Accuracy@1, lower = better)")
    print("="*60)
    print(df_comparison.to_string(index=False))
    
    # Save to CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_comparison.to_csv(output_path, index=False)
    print(f"\n✅ Results saved to: {output_path}")
    
    # Print best/worst
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    best_row = df_comparison.iloc[0]
    worst_row = df_comparison.iloc[-1]
    
    print(f"\n🏆 BEST PRIVACY (lowest Accuracy@1):")
    print(f"   {best_row['filename']}")
    print(f"   Parameters: {best_row['parameters']}")
    print(f"   Accuracy@1: {best_row['Accuracy@1']:.2f}%")
    
    print(f"\n⚠️  WORST PRIVACY (highest Accuracy@1):")
    print(f"   {worst_row['filename']}")
    print(f"   Parameters: {worst_row['parameters']}")
    print(f"   Accuracy@1: {worst_row['Accuracy@1']:.2f}%")
    
    print("\n" + "="*60)
    print("BATCH EVALUATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
