"""
Utilities for managing project configuration and data loading.
"""

import os
import random
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime

# For reproducibility
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def set_all_seeds(seed: int = 42) -> None:
    """
    Set all seeds for deterministic results.
    
    Args:
        seed: Seed value
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
    if HAS_TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    print(f"✅ Seed set to {seed}. Results are reproducible.")


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dictionary with configuration
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def get_project_root() -> Path:
    """
    Get the project root.
    
    Returns:
        Path to project root
    """
    # Go up from src directory to root
    current = Path(__file__).resolve().parent
    return current.parent


def get_dataset_paths(config: Dict[str, Any], base_dir: Optional[str] = None) -> Dict[str, Path]:
    """
    Get full paths to datasets.
    
    Args:
        config: Loaded configuration
        base_dir: Base data directory (if None, use project root)
        
    Returns:
        Dictionary with dataset paths
    """
    if base_dir is None:
        base_path = get_project_root() / "data"
    else:
        base_path = Path(base_dir)
    
    return {
        "train": base_path / config["dataset"]["train"],
        "validation": base_path / config["dataset"]["validation"],
        "test": base_path / config["dataset"]["test"]
    }


def load_dataset(
    file_path: Path,
    text_column: str = "text",
    label_column: str = "label",
    sample_size: Optional[int] = None
) -> Tuple[List[str], List[str]]:
    """
    Load a CSV dataset and return texts and labels.
    
    Args:
        file_path: Path to CSV file
        text_column: Name of text column
        label_column: Name of label column
        sample_size: Number of samples to load (None = all)
        
    Returns:
        Tuple (text_list, label_list)
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")
    
    df = pd.read_csv(file_path)
    
    if sample_size is not None:
        df = df.head(sample_size)
    
    texts = df[text_column].tolist()
    labels = df[label_column].tolist()
    
    return texts, labels


def load_all_datasets(
    config: Dict[str, Any],
    base_dir: Optional[str] = None,
    sample_sizes: Optional[Dict[str, int]] = None
) -> Dict[str, Tuple[List[str], List[str]]]:
    """
    Load all datasets (train, validation, test).
    
    Args:
        config: Loaded configuration
        base_dir: Base data directory
        sample_sizes: Dict with sample counts per split {"train": 1000, "validation": 500}
        
    Returns:
        Dictionary with loaded datasets
    """
    paths = get_dataset_paths(config, base_dir)
    text_col = config["dataset"]["text_column"]
    label_col = config["dataset"]["label_column"]
    
    datasets = {}
    
    for split_name, path in paths.items():
        if path.exists():
            sample_size = None
            if sample_sizes and split_name in sample_sizes:
                sample_size = sample_sizes[split_name]
            
            texts, labels = load_dataset(path, text_col, label_col, sample_size)
            datasets[split_name] = (texts, labels)
            print(f"✅ Loaded {split_name}: {len(texts)} samples")
        else:
            print(f"⚠️  File not found: {path}")
    
    return datasets


def create_output_dir(config: Dict[str, Any]) -> Path:
    """
    Create output directory if it doesn't exist.
    
    Args:
        config: Loaded configuration
        
    Returns:
        Path to output directory
    """
    output_dir = get_project_root() / config["output"]["results_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_anonymized_dataset(
    texts: List[str],
    labels: List[str],
    output_path: Path,
    method_name: str = "anonymized"
) -> None:
    """
    Save an anonymized dataset to file.
    
    Args:
        texts: List of anonymized texts
        labels: List of labels
        output_path: Output directory
        method_name: Name of the anonymization method used
    """
    df = pd.DataFrame({
        'text': texts,
        'label': labels
    })
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"anonymized_{method_name}_{timestamp}.csv"
    filepath = output_path / filename
    
    df.to_csv(filepath, index=False)
    print(f"💾 Saved: {filepath}")
    
    return filepath


def save_metrics(
    metrics: Dict[str, Any],
    output_path: Path,
    method_name: str = "metrics"
) -> Path:
    """
    Save metrics to file.
    
    Args:
        metrics: Dictionary with metrics
        output_path: Output directory
        method_name: Identifying name
        
    Returns:
        Path to saved file
    """
    import json
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"metrics_{method_name}_{timestamp}.json"
    filepath = output_path / filename
    
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"💾 Metrics saved: {filepath}")
    return filepath


def print_comparison_table(results: Dict[str, Dict[str, float]]) -> None:
    """
    Print a comparative table of results.
    
    Args:
        results: Dictionary {method: {metric: value}}
    """
    print("\n" + "="*80)
    print("METRICS COMPARISON")
    print("="*80)
    print(f"{'Method':<15} {'Levenshtein↓':>12} {'Jaccard↓':>12} {'Cosine↑':>12} {'NER↑':>12}")
    print("-"*80)
    
    for method, metrics in results.items():
        lev = metrics.get('levenshtein_ratio', 0)
        jac = metrics.get('jaccard_similarity', 0)
        cos = metrics.get('cosine_similarity', 0)
        ner = metrics.get('ner_score', 0)
        print(f"{method:<15} {lev:>12.4f} {jac:>12.4f} {cos:>12.4f} {ner:>12.4f}")
    
    print("="*80)
