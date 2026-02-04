"""
Sentiment Classifier Module

This module implements a three-class sentiment classifier (negative, neutral, positive)
based on fine-tuned BERT models from Hugging Face Transformers.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    AutoConfig,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import warnings
import re
import emoji

warnings.filterwarnings("ignore")


class TextDataset(Dataset):
    """PyTorch Dataset for text classification."""
    
    def __init__(self, texts, labels, tokenizer, max_len):
        """
        Initialize the dataset.
        
        Args:
            texts (pd.Series): Text samples
            labels (pd.Series): Corresponding labels
            tokenizer: Hugging Face tokenizer
            max_len (int): Maximum sequence length
        """
        self.texts = texts.reset_index(drop=True)
        self.labels = labels.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class SentimentClassifier:
    """
    Three-class sentiment classifier using fine-tuned BERT.
    
    Attributes:
        model_name (str): Hugging Face model identifier
        max_len (int): Maximum sequence length
        batch_size (int): Training batch size
        epochs (int): Number of training epochs
        device (torch.device): Device for training (CPU/GPU)
    """
    
    def __init__(self, config):
        """
        Initialize the sentiment classifier.
        
        Args:
            config (dict): Configuration dictionary with classifier parameters
        """
        # Model configuration
        self.model_name = config.get('model_name', 'cardiffnlp/twitter-roberta-base-sentiment-latest')
        self.max_len = config.get('max_len', 80)
        self.batch_size = config.get('batch_size', 64)
        self.epochs = config.get('epochs', 10)
        
        # Training hyperparameters
        self.learning_rate = config.get('learning_rate', 2e-5)
        self.warmup_ratio = config.get('warmup_ratio', 0.1)
        self.att_dropout = config.get('att_dropout', 0.2)
        self.dropout = config.get('dropout', 0.2)
        self.weight_decay = config.get('weight_decay', 0.1)
        self.label_smoothing = config.get('label_smoothing', 0.1)
        self.patience = config.get('patience', 2)
        self.clip_value = config.get('clip_value', 1.0)
        self.freeze_layers = config.get('freeze_layers', 4)
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Model components (initialized during training)
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        
        # Label mapping
        self.label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
        self.label_names = ['Negative', 'Neutral', 'Positive']
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_f1': [],
            'val_loss': [],
            'val_f1': [],
            'learning_rate': []
        }
    
    def preprocess_text(self, text):
        """
        Clean and preprocess text.
        
        Args:
            text (str): Raw text
            
        Returns:
            str: Cleaned text
        """
        # Handle null or non-string values
        if not isinstance(text, str):
            return ""
        
        # Normalize smart quotes
        text = text.replace(''', "'").replace(''', "'")
        
        # Remove emojis
        text = emoji.replace_emoji(text, replace='')
        
        # Protection vault for special elements
        vault = []
        
        def protect(match_obj):
            item = match_obj.group(0)
            vault.append(item)
            pos = len(vault) - 1
            return f" __PROTECTED_{pos}__ "
        
        # Compile regex patterns
        url_pattern = re.compile(r'\b(https?://|www\.)\S+')
        email_pattern = re.compile(r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b')
        user_pattern = re.compile(r'\B@\w+')
        hashtag_pattern = re.compile(r'\B#\w+')
        
        # Apply protection
        text = url_pattern.sub(protect, text)
        text = email_pattern.sub(protect, text)
        text = user_pattern.sub(protect, text)
        text = hashtag_pattern.sub(protect, text)
        
        # Lowercase
        text = text.lower()
        
        # Spacing around punctuation
        text = re.sub(r"([!?'\".,;():\\])", r' \1 ', text)
        
        # Replace newlines/tabs
        text = re.sub(r'[\n\t]+', ' ', text)
        
        # Collapse whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Restore protected items
        def restore(match_obj):
            index = int(match_obj.group(1))
            if index < len(vault):
                return vault[index]
            return ""
        
        restore_pattern = re.compile(r'__protected_(\d+)__')
        text = restore_pattern.sub(restore, text)
        
        # Final cleanup
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def prepare_data(self, df_train, df_val, df_test=None):
        """
        Prepare and preprocess datasets.
        
        Args:
            df_train (pd.DataFrame): Training dataframe with 'text' and 'label' columns
            df_val (pd.DataFrame): Validation dataframe
            df_test (pd.DataFrame, optional): Test dataframe
            
        Returns:
            tuple: DataLoaders for train, validation, and optionally test sets
        """
        print("Preprocessing text data...")
        
        # Apply text cleaning
        df_train['text'] = df_train['text'].apply(self.preprocess_text)
        df_val['text'] = df_val['text'].apply(self.preprocess_text)
        if df_test is not None:
            df_test['text'] = df_test['text'].apply(self.preprocess_text)
        
        # Map labels to integers
        df_train['label'] = df_train['label'].map(self.label_map)
        df_val['label'] = df_val['label'].map(self.label_map)
        if df_test is not None:
            df_test['label'] = df_test['label'].map(self.label_map)
        
        # Check for NaN values
        nan_count = df_train['label'].isna().sum()
        if nan_count > 0:
            print(f"Warning: {nan_count} NaN labels found in training data")
        
        # Extract features and labels
        X_train, y_train = df_train['text'], df_train['label']
        X_val, y_val = df_val['text'], df_val['label']
        
        print(f"Dataset sizes - Train: {len(X_train)}, Val: {len(X_val)}", end="")
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Create datasets
        train_dataset = TextDataset(X_train, y_train, self.tokenizer, self.max_len)
        val_dataset = TextDataset(X_val, y_val, self.tokenizer, self.max_len)
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            num_workers=4,
            pin_memory=True
        )
        
        if df_test is not None:
            X_test, y_test = df_test['text'], df_test['label']
            print(f", Test: {len(X_test)}")
            test_dataset = TextDataset(X_test, y_test, self.tokenizer, self.max_len)
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.batch_size,
                num_workers=4,
                pin_memory=True
            )
            return train_loader, val_loader, test_loader, y_train
        else:
            print()
            return train_loader, val_loader, y_train
    
    def build_model(self, class_weights):
        """
        Build and configure the BERT model.
        
        Args:
            class_weights (np.array): Class weights for imbalanced datasets
        """
        print("Building model...")
        
        # Load configuration and modify dropout
        config = AutoConfig.from_pretrained(self.model_name, num_labels=3)
        config.hidden_dropout_prob = self.dropout
        config.attention_probs_dropout_prob = self.att_dropout
        
        # Load model with custom configuration
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            config=config
        )
        self.model = self.model.to(self.device)
        
        # Freeze early layers
        for name, param in self.model.named_parameters():
            if "transformer.layer" in name:
                layer_num = int(name.split(".")[3])
                if layer_num < self.freeze_layers:
                    param.requires_grad = False
        
        # Count trainable parameters
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
        
        # Setup optimizer (only trainable parameters)
        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Setup loss function with class weights
        weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(self.device)
        self.criterion = nn.CrossEntropyLoss(
            weight=weights_tensor,
            label_smoothing=self.label_smoothing
        )
        
        print(f"Class weights: {class_weights}")
    
    def setup_scheduler(self, train_loader):
        """
        Setup learning rate scheduler.
        
        Args:
            train_loader (DataLoader): Training data loader
        """
        num_training_steps = len(train_loader) * self.epochs
        num_warmup_steps = int(self.warmup_ratio * num_training_steps)
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        print(f"\nScheduler Info:")
        print(f"  Total training steps: {num_training_steps:,}")
        print(f"  Warmup steps: {num_warmup_steps:,} ({self.warmup_ratio*100:.0f}%)")
        print(f"  Steps per epoch: {len(train_loader):,}")
        print(f"  Initial LR: 0 → {self.learning_rate}")
    
    def train_epoch(self, data_loader):
        """
        Train for one epoch.
        
        Args:
            data_loader (DataLoader): Training data loader
            
        Returns:
            tuple: (average loss, macro F1 score)
        """
        self.model.train()
        losses = []
        preds, true_labels = [], []
        
        loop = tqdm(data_loader, desc="Training", leave=False)
        
        for d in loop:
            input_ids = d["input_ids"].to(self.device)
            attention_mask = d["attention_mask"].to(self.device)
            labels = d["labels"].to(self.device)
            
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            loss = self.criterion(outputs.logits, labels)
            
            _, prediction = torch.max(outputs.logits, dim=1)
            preds.extend(prediction.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            
            losses.append(loss.item())
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_value)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            
            # Show current learning rate in progress bar
            current_lr = self.scheduler.get_last_lr()[0]
            loop.set_postfix(loss=loss.item(), lr=f"{current_lr:.2e}")
        
        epoch_f1 = f1_score(true_labels, preds, average='macro')
        epoch_loss = np.mean(losses)
        
        torch.cuda.empty_cache()
        return epoch_loss, epoch_f1
    
    def evaluate(self, data_loader):
        """
        Evaluate the model.
        
        Args:
            data_loader (DataLoader): Validation/test data loader
            
        Returns:
            tuple: (average loss, accuracy, macro F1 score)
        """
        self.model.eval()
        losses = []
        preds, true_labels = [], []
        
        loop = tqdm(data_loader, desc="Evaluating", leave=False)
        
        with torch.no_grad():
            for d in loop:
                input_ids = d["input_ids"].to(self.device)
                attention_mask = d["attention_mask"].to(self.device)
                labels = d["labels"].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = self.criterion(outputs.logits, labels)
                losses.append(loss.item())
                
                _, prediction = torch.max(outputs.logits, dim=1)
                preds.extend(prediction.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        torch.cuda.empty_cache()
        
        f1 = f1_score(true_labels, preds, average='macro')
        acc = accuracy_score(true_labels, preds)
        
        return np.mean(losses), acc, f1
    
    def train(self, train_loader, val_loader, output_path="best_model_sentiment.pt"):
        """
        Train the model with early stopping.
        
        Args:
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader
            output_path (str): Path to save the best model
            
        Returns:
            dict: Training history
        """
        print(f"\nStarting training for {self.epochs} epochs...")
        
        best_f1 = 0
        patience_counter = 0
        
        for epoch in range(self.epochs):
            train_loss, train_f1 = self.train_epoch(train_loader)
            val_loss, val_acc, val_f1 = self.evaluate(val_loader)
            
            # Track current learning rate
            current_lr = self.scheduler.get_last_lr()[0]
            
            self.history['train_loss'].append(train_loss)
            self.history['train_f1'].append(train_f1)
            self.history['val_loss'].append(val_loss)
            self.history['val_f1'].append(val_f1)
            self.history['learning_rate'].append(current_lr)
            
            # Print epoch results
            print(f'Epoch {epoch+1}/{self.epochs}: '
                  f'TrLoss {train_loss:.4f} | TrF1 {train_f1:.4f} || '
                  f'ValLoss {val_loss:.4f} | ValF1 {val_f1:.4f} || '
                  f'LR {current_lr:.2e}')
            
            # Save best model
            if val_f1 > best_f1:
                best_f1 = val_f1
                patience_counter = 0
                torch.save(self.model.state_dict(), output_path)
                print(f"  ✓ Saved best model (F1: {best_f1:.4f})")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= self.patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break
        
        print(f"\nTraining completed. Best Validation F1: {best_f1:.4f}")
        return self.history
    
    def get_predictions(self, data_loader):
        """
        Get predictions for a dataset.
        
        Args:
            data_loader (DataLoader): Data loader
            
        Returns:
            tuple: (predictions, true labels)
        """
        self.model.eval()
        predictions, real_values = [], []
        
        with torch.no_grad():
            for d in data_loader:
                input_ids = d["input_ids"].to(self.device)
                attention_mask = d["attention_mask"].to(self.device)
                labels = d["labels"]
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                _, preds = torch.max(outputs.logits, dim=1)
                
                predictions.extend(preds.cpu().numpy())
                real_values.extend(labels.numpy())
        
        return predictions, real_values
    
    def test(self, test_loader, model_path="best_model_sentiment.pt"):
        """
        Test the model and print classification report.
        
        Args:
            test_loader (DataLoader): Test data loader
            model_path (str): Path to the saved model
            
        Returns:
            tuple: (predictions, true labels, confusion matrix)
        """
        print("\nLoading best model for testing...")
        self.model.load_state_dict(torch.load(model_path))
        
        _, test_acc, test_f1 = self.evaluate(test_loader)
        print(f"Test Accuracy: {test_acc:.4f}, Test F1: {test_f1:.4f}")
        
        y_pred, y_true = self.get_predictions(test_loader)
        
        print("\n" + "="*50)
        print("Classification Report:")
        print("="*50)
        print(classification_report(y_true, y_pred, target_names=self.label_names))
        
        cm = confusion_matrix(y_true, y_pred)
        
        return y_pred, y_true, cm
    
    def load_model(self, model_path):
        """
        Load a saved model.
        
        Args:
            model_path (str): Path to the saved model
        """
        if self.model is None:
            # Initialize model architecture if not already done
            config = AutoConfig.from_pretrained(self.model_name, num_labels=3)
            config.hidden_dropout_prob = self.dropout
            config.attention_probs_dropout_prob = self.att_dropout
            
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                config=config
            )
            self.model = self.model.to(self.device)
        
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        print(f"Model loaded from {model_path}")
    
    def predict(self, texts):
        """
        Predict sentiment for a list of texts.
        
        Args:
            texts (list): List of text strings
            
        Returns:
            list: Predicted labels (as strings: 'negative', 'neutral', 'positive')
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() or train() first.")
        
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Preprocess texts
        cleaned_texts = [self.preprocess_text(text) for text in texts]
        
        # Tokenize
        encodings = self.tokenizer(
            cleaned_texts,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            input_ids = encodings['input_ids'].to(self.device)
            attention_mask = encodings['attention_mask'].to(self.device)
            
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            _, predictions = torch.max(outputs.logits, dim=1)
        
        # Convert to label names
        reverse_label_map = {v: k for k, v in self.label_map.items()}
        predicted_labels = [reverse_label_map[pred.item()] for pred in predictions]
        
        return predicted_labels
