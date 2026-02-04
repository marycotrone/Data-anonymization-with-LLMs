#!/usr/bin/env python3
"""
Script to run anonymization with LLM 

Usage:
    python scripts/run_llm.py --config configs/config.yaml --model gemma2:2b
    python scripts/run_llm.py --model llama3.2 --dataset validation --sample 50
"""

import argparse
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_anonymizer import OllamaAnonymizer
from metrics import AnonymizationMetrics
from utils import (
    load_config, 
    load_all_datasets, 
    create_output_dir,
    set_all_seeds,
    save_anonymized_dataset,
    save_metrics
)


def main():
    parser = argparse.ArgumentParser(description="Run LLM anonymization")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Ollama model name (override config)"
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
    parser.add_argument(
        "--prompt-style",
        type=str,
        choices=["paraphrase", "simple", "strict"],
        default="paraphrase",
        help="Prompt style to use"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    print("📋 Loading configuration...")
    config = load_config(args.config)
    
    # Set seeds
    set_all_seeds(config["general"]["seed"])
    
    # Override model if specified
    model_name = args.model or config["llm"]["model_name"]
    
    # Load dataset
    print(f"\nLoading dataset: {args.dataset}...")
    sample_sizes = {args.dataset: args.sample} if args.sample else None
    datasets = load_all_datasets(config, sample_sizes=sample_sizes)
    
    if args.dataset not in datasets:
        print(f"Dataset {args.dataset} not found!")
        sys.exit(1)
    
    sentences, labels = datasets[args.dataset]
    print(f"   Loaded {len(sentences)} samples")
    
    # Create output directory
    output_dir = create_output_dir(config)
    
    # Initialize LLM
    print("\n" + "="*60)
    print("LLM INITIALIZATION")
    print("="*60)
    
    llm = OllamaAnonymizer(
        model_name=model_name,
        base_url=config["llm"]["base_url"],
        temperature=config["llm"]["temperature"],
        max_tokens=config["llm"]["max_tokens"],
        prompt_style=args.prompt_style,
        verbose=config["general"]["verbose"]
    )
    
    # Show model info
    model_info = llm.get_model_info()
    if model_info and config["general"]["verbose"]:
        print(f"\nModel: {model_name}")
        if "details" in model_info:
            details = model_info["details"]
            print(f"   Family: {details.get('family', 'N/A')}")
            print(f"   Parameters: {details.get('parameter_size', 'N/A')}")
    
    # Execute anonymization
    print(f"\nAnonymizing with {model_name}...")
    anonymized_llm = llm.anonymize_batch(
        sentences,
        labels=labels,
        show_progress=config["general"]["show_progress"]
    )
    
    # Evaluate
    print("\nEvaluating LLM...")
    metrics = AnonymizationMetrics(
        sbert_model=config["metrics"]["sbert_model"],
        spacy_model=config["metrics"]["spacy_model"],
        verbose=config["general"]["verbose"]
    )
    
    llm_metrics = metrics.evaluate_all(sentences, anonymized_llm)
    
    # Save results
    if config["output"]["save_anonymized"]:
        model_safe = model_name.replace(':', '_').replace('/', '_')
        save_anonymized_dataset(anonymized_llm, labels, output_dir, f'llm_{model_safe}_{args.dataset}')
    
    # Save metrics
    if config["output"]["save_metrics"]:
        model_safe = model_name.replace(':', '_').replace('/', '_')
        save_metrics({'LLM': llm_metrics}, output_dir, f'llm_{model_safe}_{args.dataset}')
    
    # Show examples
    n_examples = config['general'].get('n_examples', 5)
    print(f"\nExamples ({n_examples} samples):")
    for i in range(min(n_examples, len(sentences))):
        print(f"\n{i+1}. Original: {sentences[i]}")
        print(f"   LLM:       {anonymized_llm[i]}")
    
    print("\nCompleted!")


if __name__ == "__main__":
    main()
