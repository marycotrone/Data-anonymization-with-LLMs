"""
Training script for the three-class sentiment classifier.

This script trains a BERT-based sentiment classifier on the provided dataset
and evaluates it on validation and test sets.

Usage:
    python scripts/train_sentiment_classifier.py [--config path/to/config.yaml]
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from sentiment_classifier import SentimentClassifier
from utils import load_config
from sklearn.utils.class_weight import compute_class_weight


def plot_training_history(history, output_dir="results"):
    """
    Plot training history.
    
    Args:
        history (dict): Training history dictionary
        output_dir (str): Directory to save plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss plot
    axes[0, 0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0, 0].plot(history['val_loss'], label='Val Loss', marker='s')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # F1 score plot
    axes[0, 1].plot(history['train_f1'], label='Train F1', marker='o')
    axes[0, 1].plot(history['val_f1'], label='Val F1', marker='s')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('F1 Score')
    axes[0, 1].set_title('Training and Validation F1 Score')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Learning rate plot
    axes[1, 0].plot(history['learning_rate'], marker='o', color='purple')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_title('Learning Rate Schedule')
    axes[1, 0].grid(True)
    axes[1, 0].set_yscale('log')
    
    # Hide the empty subplot
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path / 'training_history.png', dpi=300, bbox_inches='tight')
    print(f"Training history plot saved to {output_path / 'training_history.png'}")
    plt.close()


def plot_confusion_matrix(cm, labels, output_dir="results"):
    """
    Plot confusion matrix.
    
    Args:
        cm (np.array): Confusion matrix
        labels (list): Class labels
        output_dir (str): Directory to save plot
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels, cbar=True)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix (3 Classes)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {output_path / 'confusion_matrix.png'}")
    plt.close()


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train sentiment classifier')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--train-data', type=str, required=True,
                        help='Path to training data CSV')
    parser.add_argument('--val-data', type=str, required=True,
                        help='Path to validation data CSV')
    parser.add_argument('--test-data', type=str, default=None,
                        help='Path to test data CSV (optional)')
    parser.add_argument('--output', type=str, default='best_model_sentiment.pt',
                        help='Path to save the best model')
    parser.add_argument('--results-dir', type=str, default='results',
                        help='Directory to save results')
    
    args = parser.parse_args()
    
    # Load configuration
    print("Loading configuration...")
    config = load_config(args.config)
    classifier_config = config.get('sentiment_classifier', {})
    
    # Load datasets
    print("\nLoading datasets...")
    df_train = pd.read_csv(args.train_data)
    df_val = pd.read_csv(args.val_data)
    
    print(f"Training set: {len(df_train)} samples")
    print(f"Validation set: {len(df_val)} samples")
    
    # Check for required columns
    required_cols = ['text', 'label']
    for col in required_cols:
        if col not in df_train.columns:
            raise ValueError(f"Training data missing required column: {col}")
        if col not in df_val.columns:
            raise ValueError(f"Validation data missing required column: {col}")
    
    # Print class distribution
    print("\n--- Training Set Class Distribution ---")
    print(df_train['label'].value_counts())
    print("\n--- Validation Set Class Distribution ---")
    print(df_val['label'].value_counts())
    
    # Load test data if provided
    df_test = None
    if args.test_data:
        df_test = pd.read_csv(args.test_data)
        print(f"Test set: {len(df_test)} samples")
        print("\n--- Test Set Class Distribution ---")
        print(df_test['label'].value_counts())
    
    # Initialize classifier
    print("\n" + "="*60)
    print("INITIALIZING SENTIMENT CLASSIFIER")
    print("="*60)
    classifier = SentimentClassifier(classifier_config)
    
    # Prepare data
    print("\n" + "="*60)
    print("PREPARING DATA")
    print("="*60)
    if df_test is not None:
        train_loader, val_loader, test_loader, y_train = classifier.prepare_data(
            df_train, df_val, df_test
        )
    else:
        train_loader, val_loader, y_train = classifier.prepare_data(df_train, df_val)
        test_loader = None
    
    # Compute class weights
    print("\nComputing class weights for balanced training...")
    label_map = classifier.label_map
    
    # Convert string labels to integers for class weight computation
    y_train_numeric = df_train['label'].map(label_map).to_numpy()
    
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train_numeric),
        y=y_train_numeric
    )
    
    # Build model
    print("\n" + "="*60)
    print("BUILDING MODEL")
    print("="*60)
    classifier.build_model(class_weights)
    
    # Setup scheduler
    classifier.setup_scheduler(train_loader)
    
    # Train model
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)
    history = classifier.train(train_loader, val_loader, output_path=args.output)
    
    # Plot training history
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)
    plot_training_history(history, output_dir=args.results_dir)
    
    # Test model if test data provided
    if test_loader is not None:
        print("\n" + "="*60)
        print("TESTING")
        print("="*60)
        y_pred, y_true, cm = classifier.test(test_loader, model_path=args.output)
        
        # Plot confusion matrix
        plot_confusion_matrix(cm, classifier.label_names, output_dir=args.results_dir)
        
        # Save predictions
        results_df = pd.DataFrame({
            'true_label': [classifier.label_names[y] for y in y_true],
            'predicted_label': [classifier.label_names[y] for y in y_pred],
            'correct': [yt == yp for yt, yp in zip(y_true, y_pred)]
        })
        
        results_path = Path(args.results_dir) / 'test_predictions.csv'
        results_df.to_csv(results_path, index=False)
        print(f"Test predictions saved to {results_path}")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Best model saved to: {args.output}")
    print(f"Results saved to: {args.results_dir}/")


if __name__ == "__main__":
    main()
