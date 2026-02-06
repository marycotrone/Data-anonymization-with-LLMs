"""
Evaluation metrics for text anonymization.

Includes metrics for:
- Irreversibility: Levenshtein Ratio, Jaccard Similarity
- Utility: Cosine Similarity (with SBERT)
- Anonymization: NER Score
- Privacy: Paraphrase Retrieval (Accuracy@k)
"""

import re
import numpy as np
import torch
from typing import List, Tuple, Dict, Optional
from nltk.tokenize import word_tokenize
from nltk.metrics.distance import edit_distance
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class AnonymizationMetrics:
    """Class for calculating anonymization evaluation metrics."""
    
    def __init__(
        self,
        sbert_model: str = "intfloat/e5-large-v2",
        spacy_model: str = "en_core_web_sm",
        verbose: bool = True
    ):
        """
        Initialize the metrics.
        
        Args:
            sbert_model: Sentence-BERT model for cosine similarity
            spacy_model: Spacy model for NER
            verbose: Show loading messages
        """
        self.verbose = verbose
        
        # Load SBERT
        if self.verbose:
            print(f"Loading SBERT: {sbert_model}...")
        self.sbert = SentenceTransformer(sbert_model)
        
        # Load Spacy
        if self.verbose:
            print(f"Loading Spacy: {spacy_model}...")
        self.nlp = spacy.load(spacy_model)
        
        if self.verbose:
            print("✅ Models loaded")
    
    @staticmethod
    def clean_text_for_ner(text: str) -> str:
        """
        Clean text to improve entity recognition.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        # Remove @ and # keeping the text
        text = re.sub(r'[@#]', '', text)
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        # Normalize spaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def calculate_levenshtein_ratio(
        self,
        original_sentences: List[str],
        generated_sentences: List[str]
    ) -> float:
        """
        Calculate word-level Levenshtein Ratio.
        
        Measures irreversibility: lower is better.
        
        Args:
            original_sentences: Original sentences
            generated_sentences: Generated sentences
            
        Returns:
            Average Levenshtein Ratio (0-1)
        """
        scores = self.calculate_levenshtein_ratio_list(original_sentences, generated_sentences)
        return np.mean(scores) if len(scores) > 0 else 0
    
    def calculate_levenshtein_ratio_list(
        self,
        original_sentences: List[str],
        generated_sentences: List[str]
    ) -> List[float]:
        """
        Calculate word-level Levenshtein Ratio for each sentence pair.
        
        Args:
            original_sentences: Original sentences
            generated_sentences: Generated sentences
            
        Returns:
            List of Levenshtein Ratio scores (0-1) for each pair
        """
        scores = []
        
        for orig, gen in zip(original_sentences, generated_sentences):
            orig_tokens = word_tokenize(orig.lower())
            gen_tokens = word_tokenize(gen.lower())
            
            # Handle special cases
            if not orig_tokens and not gen_tokens:
                scores.append(1.0)
                continue
            if not orig_tokens or not gen_tokens:
                scores.append(0.0)
                continue
            
            # Calculate distance
            distance = edit_distance(orig_tokens, gen_tokens)
            max_len = max(len(orig_tokens), len(gen_tokens))
            ratio = 1.0 - (distance / max_len)
            scores.append(ratio)
        
        return scores
    
    def calculate_jaccard_similarity(
        self,
        original_sentences: List[str],
        generated_sentences: List[str]
    ) -> float:
        """
        Calculate Jaccard Similarity.
        
        Measures irreversibility: lower is better.
        
        Args:
            original_sentences: Original sentences
            generated_sentences: Generated sentences
            
        Returns:
            Average Jaccard Similarity (0-1)
        """
        scores = self.calculate_jaccard_similarity_list(original_sentences, generated_sentences)
        return np.mean(scores) if len(scores) > 0 else 0
    
    def calculate_jaccard_similarity_list(
        self,
        original_sentences: List[str],
        generated_sentences: List[str]
    ) -> List[float]:
        """
        Calculate Jaccard Similarity for each sentence pair.
        
        Args:
            original_sentences: Original sentences
            generated_sentences: Generated sentences
            
        Returns:
            List of Jaccard Similarity scores (0-1) for each pair
        """
        scores = []
        
        for orig, gen in zip(original_sentences, generated_sentences):
            orig_set = set(word_tokenize(orig.lower()))
            gen_set = set(word_tokenize(gen.lower()))
            
            # Handle special cases
            if not orig_set and not gen_set:
                scores.append(1.0)
                continue
            if not orig_set or not gen_set:
                scores.append(0.0)
                continue
            
            # Calculate Jaccard
            intersection = len(orig_set.intersection(gen_set))
            union = len(orig_set.union(gen_set))
            scores.append(intersection / union)
        
        return scores
    
    def calculate_cosine_similarity(
        self,
        original_sentences: List[str],
        generated_sentences: List[str],
        show_progress: bool = True
    ) -> float:
        """
        Calculate Cosine Similarity with SBERT embeddings.
        
        Measures semantic utility: higher is better.
        
        Args:
            original_sentences: Original sentences
            generated_sentences: Generated sentences
            show_progress: Show progress bar
            
        Returns:
            Average Cosine Similarity (0-1)
        """
        similarities = self.calculate_cosine_similarity_list(
            original_sentences, generated_sentences, show_progress
        )
        return np.mean(similarities) if len(similarities) > 0 else 0
    
    def calculate_cosine_similarity_list(
        self,
        original_sentences: List[str],
        generated_sentences: List[str],
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Calculate Cosine Similarity for each sentence pair.
        
        Args:
            original_sentences: Original sentences
            generated_sentences: Generated sentences
            show_progress: Show progress bar
            
        Returns:
            Array of Cosine Similarity scores (0-1) for each pair
        """
        if len(original_sentences) != len(generated_sentences):
            raise ValueError("Lists must have the same length")
        
        # Generate embeddings
        orig_embeddings = self.sbert.encode(
            original_sentences,
            show_progress_bar=show_progress
        )
        gen_embeddings = self.sbert.encode(
            generated_sentences,
            show_progress_bar=show_progress
        )
        
        # Calculate similarity for each pair
        similarities = np.diag(cosine_similarity(orig_embeddings, gen_embeddings))
        
        return similarities
    
    def calculate_ner_score(
        self,
        original_sentences: List[str],
        generated_sentences: List[str],
        strict_mode: bool = True,
        target_labels: List[str] = None
    ) -> float:
        """
        Calculate NER Score to measure anonymization.
        
        0.0 = Failure (all entities visible)
        1.0 = Success (no original entities found)
        
        Args:
            original_sentences: Original sentences
            generated_sentences: Generated sentences
            strict_mode: If True, check individual words in multi-word entities
            target_labels: NER labels to consider (default: PERSON, GPE, ORG, LOC)
            
        Returns:
            Average NER Score (0-1)
        """
        scores = self.calculate_ner_score_list(
            original_sentences, generated_sentences, strict_mode, target_labels
        )
        return np.mean(scores) if len(scores) > 0 else 1.0
    
    def calculate_ner_score_list(
        self,
        original_sentences: List[str],
        generated_sentences: List[str],
        strict_mode: bool = True,
        target_labels: List[str] = None
    ) -> List[float]:
        """
        Calculate NER Score for each sentence pair.
        
        Args:
            original_sentences: Original sentences
            generated_sentences: Generated sentences
            strict_mode: If True, check individual words in multi-word entities
            target_labels: NER labels to consider (default: PERSON, GPE, ORG, LOC)
            
        Returns:
            List of NER scores (0-1) for each pair
        """
        if target_labels is None:
            target_labels = ["PERSON", "GPE", "ORG", "LOC"]
        
        scores = []
        
        # Clean the texts
        orig_cleaned = [self.clean_text_for_ner(s) for s in original_sentences]
        gen_cleaned = [self.clean_text_for_ner(s) for s in generated_sentences]
        
        # Disable unnecessary Spacy components
        with self.nlp.select_pipes(disable=["parser", "tagger", "lemmatizer"]):
            for orig, gen in zip(orig_cleaned, gen_cleaned):
                doc_orig = self.nlp(orig)
                
                # Filter target entities
                orig_ents = [
                    ent for ent in doc_orig.ents
                    if ent.label_ in target_labels
                ]
                
                if not orig_ents:
                    continue
                
                gen_text_lower = gen.lower()
                leaked_entities = 0
                
                for ent in orig_ents:
                    entity_text = ent.text.lower()
                    is_leaked = False
                    
                    if strict_mode:
                        # Check individual words as well
                        tokens = entity_text.split()
                        if any(t in gen_text_lower for t in tokens if len(t) > 2):
                            is_leaked = True
                    else:
                        # Check only exact string
                        if entity_text in gen_text_lower:
                            is_leaked = True
                    
                    if is_leaked:
                        leaked_entities += 1
                
                # Calculate score
                score = 1.0 - (leaked_entities / len(orig_ents))
                scores.append(score)
        
        return scores
    
    def evaluate_all(
        self,
        original_sentences: List[str],
        generated_sentences: List[str],
        show_progress: bool = True
    ) -> dict:
        """
        Calculate all metrics simultaneously.
        
        Args:
            original_sentences: Original sentences
            generated_sentences: Generated sentences
            show_progress: Show progress
            
        Returns:
            Dictionary with all metrics
        """
        if self.verbose:
            print("Calculating evaluation metrics...")
        
        results = {
            "levenshtein_ratio": self.calculate_levenshtein_ratio(
                original_sentences, generated_sentences
            ),
            "jaccard_similarity": self.calculate_jaccard_similarity(
                original_sentences, generated_sentences
            ),
            "cosine_similarity": self.calculate_cosine_similarity(
                original_sentences, generated_sentences, show_progress
            ),
            "ner_score": self.calculate_ner_score(
                original_sentences, generated_sentences
            )
        }
        
        if self.verbose:
            print("\n=== RESULTS ===")
            print(f"Levenshtein Ratio (↓): {results['levenshtein_ratio']:.4f}")
            print(f"Jaccard Similarity (↓): {results['jaccard_similarity']:.4f}")
            print(f"Cosine Similarity (↑): {results['cosine_similarity']:.4f}")
            print(f"NER Score (↑): {results['ner_score']:.4f}")
        
        return results
    
    def evaluate_paraphrase_retrieval(
        self,
        original_sentences: List[str],
        paraphrased_sentences: List[str],
        k_values: List[int] = [1, 5, 10],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate paraphrase retrieval performance.
        
        This metric measures privacy: how easy is it to retrieve the original
        sentence from the paraphrased version? Lower accuracy = better privacy.
        
        For each paraphrased sentence, we search for the most similar sentence
        in the original corpus. If the correct original sentence is found in
        the top-k results, we count it as a match.
        
        Args:
            original_sentences: List of original sentences (corpus)
            paraphrased_sentences: List of paraphrased sentences (queries)
            k_values: List of k values for Accuracy@k (default: [1, 5, 10])
            batch_size: Batch size for encoding (not currently used, kept for compatibility)
            show_progress: Show progress bars
            
        Returns:
            Dictionary with accuracy metrics for each k value
            Example: {"Accuracy@1": 45.2, "Accuracy@5": 78.3, "Accuracy@10": 89.1}
        """
        # Ensure inputs are lists
        if hasattr(original_sentences, 'tolist'):
            original_sentences = original_sentences.tolist()
        if hasattr(paraphrased_sentences, 'tolist'):
            paraphrased_sentences = paraphrased_sentences.tolist()
        
        if self.verbose:
            print(f"\n{'='*60}")
            print("PARAPHRASE RETRIEVAL EVALUATION")
            print(f"{'='*60}")
            print(f"Original corpus size: {len(original_sentences)}")
            print(f"Paraphrased queries: {len(paraphrased_sentences)}")
        
        # Step 1: Encode all original sentences (create search index)
        if show_progress and self.verbose:
            print(f"\n1. Creating vector index for {len(original_sentences)} original sentences...")
        
        original_embeddings = self.sbert.encode(
            original_sentences,
            convert_to_tensor=True,
            show_progress_bar=show_progress and self.verbose
        )
        
        # Step 2: Encode paraphrased sentences
        if show_progress and self.verbose:
            print(f"\n2. Encoding {len(paraphrased_sentences)} paraphrased sentences...")
        
        paraphrase_embeddings = self.sbert.encode(
            paraphrased_sentences,
            convert_to_tensor=True,
            show_progress_bar=show_progress and self.verbose
        )
        
        # Step 3: Calculate similarity and ranking (Retrieval)
        if show_progress and self.verbose:
            print("\n3. Computing similarity and ranking (Retrieval)...")
        
        # Initialize counters for each k value
        correct_counts = {k: 0 for k in k_values}
        total = len(paraphrased_sentences)
        max_k = max(k_values)
        
        # Iterate over each paraphrase
        # ASSUMPTION: paraphrased_sentences[i] is the paraphrase of original_sentences[i]
        iterator = range(total)
        if show_progress:
            iterator = tqdm(iterator, desc="Evaluating retrieval")
        
        for i in iterator:
            query_emb = paraphrase_embeddings[i]
            
            # Calculate cosine similarity between current paraphrase and ALL original sentences
            cos_scores = util.cos_sim(query_emb, original_embeddings)[0]
            
            # Find top-k indices with highest scores
            top_results = torch.topk(cos_scores, k=max_k)
            top_indices = top_results.indices.tolist()
            
            # Check if correct index (ground truth = i) is in top-k for each k
            for k in k_values:
                if i in top_indices[:k]:
                    correct_counts[k] += 1
        
        # Calculate accuracy percentages
        metrics = {
            f"Accuracy@{k}": (correct_counts[k] / total) * 100
            for k in k_values
        }
        
        # Print results
        if self.verbose:
            print(f"\n{'='*60}")
            print("RETRIEVAL RESULTS")
            print(f"{'='*60}")
            for k in k_values:
                print(f"Accuracy@{k:2d}: {metrics[f'Accuracy@{k}']:6.2f}%")
            print(f"{'='*60}\n")
        
        return metrics
    
    def plot_metric_distributions(
        self,
        scores_dict: Dict[str, List[float]],
        metric_name: str,
        color_palette: str = "viridis",
        save_path: Optional[str] = None,
        plot_type: str = "ridge"
    ) -> None:
        """
        Plot distribution of metric scores across different datasets/configs.
        
        Two visualization types available:
        - 'ridge': Ridge plot (stacked density plots)
        - 'overlay': Overlapping histograms
        
        Args:
            scores_dict: Dictionary mapping dataset labels to score lists
                        Example: {"T=0.3 P=0.7": [0.8, 0.75, ...], ...}
            metric_name: Name of the metric for the plot title
            color_palette: Seaborn color palette name
            save_path: Optional path to save the plot
            plot_type: Type of plot ('ridge' or 'overlay')
        """
        if plot_type == "ridge":
            self._plot_ridge(scores_dict, metric_name, color_palette, save_path)
        elif plot_type == "overlay":
            self._plot_overlay(scores_dict, metric_name, color_palette, save_path)
        else:
            raise ValueError(f"Unknown plot_type: {plot_type}. Use 'ridge' or 'overlay'.")
    
    def _plot_ridge(
        self,
        scores_dict: Dict[str, List[float]],
        metric_name: str,
        color_palette: str,
        save_path: Optional[str]
    ) -> None:
        """
        Create a ridge plot (stacked density plots) for metric distributions.
        
        Args:
            scores_dict: Dictionary mapping labels to score lists
            metric_name: Metric name for title
            color_palette: Seaborn palette
            save_path: Optional save path
        """
        # Prepare data
        all_data = []
        for label, scores in scores_dict.items():
            if hasattr(scores, 'tolist'):
                scores = scores.tolist()
            for score in scores:
                all_data.append({"Score": score, "Dataset": label})
        
        df_plot = pd.DataFrame(all_data)
        
        # Configure aesthetics
        sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
        
        # Create FacetGrid
        n_datasets = len(scores_dict.keys())
        g = sns.FacetGrid(
            df_plot, row="Dataset", hue="Dataset",
            aspect=10, height=0.8, palette=color_palette
        )
        
        # Draw density plots
        g.map(sns.kdeplot, "Score", bw_adjust=0.5, clip_on=False, fill=True, alpha=0.7, linewidth=1.5)
        g.map(sns.kdeplot, "Score", clip_on=False, color="w", lw=2, bw_adjust=0.5)
        g.map(plt.axhline, y=0, lw=2, clip_on=False, color="grey", alpha=0.3)
        
        # Add labels
        def label(x, color, label):
            ax = plt.gca()
            ax.text(0, 0.2, label, fontweight="bold", color=color,
                   ha="left", va="center", transform=ax.transAxes)
        
        g.map(label, "Score")
        
        # Cleanup
        g.set_titles("")
        g.set(yticks=[], ylabel="")
        g.despine(bottom=True, left=True)
        
        plt.subplots_adjust(hspace=-0.25)
        g.fig.suptitle(f"Ridge Plot: {metric_name}", fontsize=16, fontweight='bold', y=0.98)
        plt.xlabel("Score", fontsize=12)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if self.verbose:
                print(f"Plot saved: {save_path}")
        
        plt.show()
    
    def _plot_overlay(
        self,
        scores_dict: Dict[str, List[float]],
        metric_name: str,
        color_palette: str,
        save_path: Optional[str]
    ) -> None:
        """
        Create an overlapping histogram plot for metric distributions.
        
        Args:
            scores_dict: Dictionary mapping labels to score lists
            metric_name: Metric name for title
            color_palette: Seaborn palette
            save_path: Optional save path
        """
        # Prepare data
        all_data = []
        for label, scores in scores_dict.items():
            if hasattr(scores, 'tolist'):
                scores = scores.tolist()
            for score in scores:
                all_data.append({"Score": score, "Dataset": label})
        
        df_plot = pd.DataFrame(all_data)
        
        # Create plot
        plt.figure(figsize=(10, 6))
        
        sns.histplot(
            data=df_plot,
            x="Score",
            hue="Dataset",
            kde=True,
            element="step",
            palette=color_palette,
            bins=30,
            alpha=0.5,
            common_norm=False
        )
        
        plt.title(f"Distribution Comparison: {metric_name}", fontsize=14, fontweight='bold')
        plt.xlabel("Score", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if self.verbose:
                print(f"Plot saved: {save_path}")
        
        plt.show()
