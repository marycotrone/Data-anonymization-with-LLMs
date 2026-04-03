import re
import numpy as np
import torch
from typing import List, Dict, Optional
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer, util


class RetrievalEvaluator:
    """Unified class for evaluating privacy through retrieval attacks."""
    
    def __init__(
        self,
        sbert_model: str = "intfloat/e5-large-v2",
        verbose: bool = True
    ):
        """
        Initialize the retrieval evaluator.
        
        Args:
            sbert_model: Sentence-BERT model for semantic similarity
            verbose: Show loading messages and progress
        """
        self.verbose = verbose
        
        # Load SBERT for semantic retrieval
        if self.verbose:
            print(f"Loading SBERT: {sbert_model}...")
        self.sbert = SentenceTransformer(sbert_model)
        
        # Try to import rapidfuzz for Levenshtein
        try:
            from rapidfuzz.distance import Levenshtein
            from rapidfuzz.process import extract
            self.Levenshtein = Levenshtein
            self.extract = extract
            self.rapidfuzz_available = True
        except ImportError:
            if self.verbose:
                print("⚠️  rapidfuzz not installed. Levenshtein retrieval will not be available.")
                print("   Install with: pip install rapidfuzz")
            self.rapidfuzz_available = False
        
        if self.verbose:
            print("✅ RetrievalEvaluator initialized")
    
    def evaluate_sbert_retrieval(
        self,
        original_sentences: List[str],
        paraphrased_sentences: List[str],
        k_values: List[int] = [1, 5, 10],
        show_progress: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate retrieval using SBERT semantic similarity.
        
        Args:
            original_sentences: List of original sentences (corpus)
            paraphrased_sentences: List of paraphrased sentences (queries)
            k_values: List of k values for Accuracy@k
            show_progress: Show progress bars
            
        Returns:
            Dictionary with accuracy metrics: {"Accuracy@1": ..., "Accuracy@5": ..., ...}
        """
        if self.verbose:
            print("\n" + "="*70)
            print("SBERT RETRIEVAL (Semantic Similarity)")
            print("="*70)
        
        # Encode original sentences
        original_embeddings = self.sbert.encode(
            original_sentences,
            convert_to_tensor=True,
            show_progress_bar=show_progress
        )
        
        # Encode paraphrased sentences
        paraphrase_embeddings = self.sbert.encode(
            paraphrased_sentences,
            convert_to_tensor=True,
            show_progress_bar=show_progress
        )
        
        # Calculate retrieval accuracy
        correct_counts = {k: 0 for k in k_values}
        total = len(paraphrased_sentences)
        max_k = min(max(k_values), len(original_sentences))
        
        iterator = tqdm(range(total), desc="SBERT Retrieval", leave=False) if show_progress else range(total)
        
        for i in iterator:
            query_emb = paraphrase_embeddings[i]
            cos_scores = util.cos_sim(query_emb, original_embeddings)[0]
            top_results = torch.topk(cos_scores, k=max_k)
            top_indices = top_results.indices.tolist()
            
            for k in k_values:
                if i in top_indices[:k]:
                    correct_counts[k] += 1
        
        return {f"Accuracy@{k}": (correct_counts[k] / total) * 100 for k in k_values}
    
    def evaluate_jaccard_retrieval(
        self,
        original_sentences: List[str],
        paraphrased_sentences: List[str],
        k_values: List[int] = [1, 5, 10],
        show_progress: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate retrieval using Jaccard similarity (word-level).
        
        Args:
            original_sentences: List of original sentences (corpus)
            paraphrased_sentences: List of paraphrased sentences (queries)
            k_values: List of k values for Accuracy@k
            show_progress: Show progress bars
            
        Returns:
            Dictionary with accuracy metrics
        """
        if self.verbose:
            print("\n" + "="*70)
            print("JACCARD RETRIEVAL (Word-level Similarity)")
            print("="*70)
        
        # Pre-process original sentences into word sets
        original_sets = [set(re.findall(r'\w+', str(s).lower())) for s in original_sentences]
        
        total = len(paraphrased_sentences)
        correct_counts = {k: 0 for k in k_values}
        max_k = min(max(k_values), len(original_sentences))

        iterator = tqdm(paraphrased_sentences, desc="Jaccard Retrieval", leave=False) if show_progress else paraphrased_sentences
        
        for i, para_sentence in enumerate(iterator):
            # Convert paraphrase to word set
            query_set = set(re.findall(r'\w+', str(para_sentence).lower()))
            
            # Calculate Jaccard similarity with all original sentences
            scores = []
            for orig_set in original_sets:
                intersection = len(query_set.intersection(orig_set))
                union = len(query_set.union(orig_set))
                score = intersection / union if union > 0 else 0
                scores.append(score)
            
            # Get top-k indices
            top_indices = np.argsort(scores)[::-1][:max_k]
            
            # Check if original index is in top-k
            for k in k_values:
                if i in top_indices[:k]:
                    correct_counts[k] += 1
        
        return {f"Accuracy@{k}": (correct_counts[k] / total) * 100 for k in k_values}
    
    def evaluate_levenshtein_retrieval(
        self,
        original_sentences: List[str],
        paraphrased_sentences: List[str],
        k_values: List[int] = [1, 5, 10],
        show_progress: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate retrieval using Levenshtein distance (character-level).
        
        Args:
            original_sentences: List of original sentences (corpus)
            paraphrased_sentences: List of paraphrased sentences (queries)
            k_values: List of k values for Accuracy@k
            show_progress: Show progress bars
            
        Returns:
            Dictionary with accuracy metrics
        """
        if not self.rapidfuzz_available:
            raise ImportError(
                "rapidfuzz is required for Levenshtein retrieval. "
                "Install with: pip install rapidfuzz"
            )
        
        if self.verbose:
            print("\n" + "="*70)
            print("LEVENSHTEIN RETRIEVAL (Character-level Edit Distance)")
            print("="*70)
            print("⚠️  Note: This method is computationally intensive.\n")
        
        def clean_text(text):
            """Remove punctuation and convert to lowercase"""
            text = str(text).lower()
            text = re.sub(r'[^\w\s]', '', text)
            text = re.sub(r'\s+', ' ', text).strip()
            return text
        
        # Pre-process both lists
        orig_list = [clean_text(s) for s in original_sentences]
        para_list = [clean_text(s) for s in paraphrased_sentences]
        
        total = len(para_list)
        correct_counts = {k: 0 for k in k_values}
        max_k = min(max(k_values), len(orig_list))

        iterator = tqdm(para_list, desc="Levenshtein Retrieval", leave=False) if show_progress else para_list
        
        for i, query in enumerate(iterator):
            # Use rapidfuzz to find most similar sentences
            results = self.extract(
                query,
                orig_list,
                scorer=self.Levenshtein.normalized_similarity,
                limit=max_k
            )
            
            # Extract indices of top matches
            top_indices = [r[2] for r in results]
            
            # Check if original index is in top-k
            for k in k_values:
                if i in top_indices[:k]:
                    correct_counts[k] += 1
        
        return {f"Accuracy@{k}": (correct_counts[k] / total) * 100 for k in k_values}
    
    def evaluate_all(
        self,
        original_sentences: List[str],
        paraphrased_sentences: List[str],
        k_values: List[int] = [1, 5, 10],
        methods: Optional[List[str]] = None,
        show_progress: bool = True
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all available retrieval methods.
        
        Args:
            original_sentences: List of original sentences (corpus)
            paraphrased_sentences: List of paraphrased sentences (queries)
            k_values: List of k values for Accuracy@k
            methods: List of methods to use. Options: ["sbert", "jaccard", "levenshtein"]
                    If None, uses all available methods.
            show_progress: Show progress bars
            
        Returns:
            Dictionary with results for each method:
            {
                "SBERT": {"Accuracy@1": ..., "Accuracy@5": ..., ...},
                "Jaccard": {...},
                "Levenshtein": {...}
            }
        """
        if methods is None:
            methods = ["sbert", "jaccard"]
            if self.rapidfuzz_available:
                methods.append("levenshtein")
        
        results = {}
        
        if "sbert" in methods:
            results["SBERT"] = self.evaluate_sbert_retrieval(
                original_sentences, paraphrased_sentences, k_values, show_progress
            )
        
        if "jaccard" in methods:
            results["Jaccard"] = self.evaluate_jaccard_retrieval(
                original_sentences, paraphrased_sentences, k_values, show_progress
            )
        
        if "levenshtein" in methods:
            if not self.rapidfuzz_available:
                if self.verbose:
                    print("\n⚠️  Skipping Levenshtein retrieval (rapidfuzz not installed)")
            else:
                results["Levenshtein"] = self.evaluate_levenshtein_retrieval(
                    original_sentences, paraphrased_sentences, k_values, show_progress
                )
        
        return results
