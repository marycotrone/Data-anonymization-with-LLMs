"""
Data Anonymization with LLMs

Modular framework for text anonymization using:
- EDA (Easy Data Augmentation)
- KNEO (Knowledge-based NEighbor Operation)
- LLM (Large Language Models via Ollama)

"""

from .eda_anonymizer import EDAAnonymizer
from .kneo_anonymizer import KNEOAnonymizer
from .llm_anonymizer import OllamaAnonymizer, create_anonymizer_from_config, PROMPT_TEMPLATES
from .metrics import AnonymizationMetrics
from .utils import (
    load_config,
    load_dataset,
    load_all_datasets,
    set_all_seeds,
    create_output_dir,
    save_anonymized_dataset,
    save_metrics,
    print_comparison_table,
    get_project_root
)

__version__ = "1.0.0"
__author__ = "Mariarosaria Cotrone"

__all__ = [
    # Anonymizers
    "EDAAnonymizer",
    "KNEOAnonymizer", 
    "OllamaAnonymizer",
    "create_anonymizer_from_config",
    "PROMPT_TEMPLATES",
    
    # Metrics
    "AnonymizationMetrics",
    
    # Utils
    "load_config",
    "load_dataset",
    "load_all_datasets",
    "set_all_seeds",
    "create_output_dir",
    "save_anonymized_dataset",
    "save_metrics",
    "print_comparison_table",
    "get_project_root",
]
