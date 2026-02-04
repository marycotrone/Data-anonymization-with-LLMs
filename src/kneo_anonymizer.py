"""
Knowledge-based NEighbor Operation (KNEO) for text anonymization.

This module implements KNEO using embeddings (GloVe, fastText) to 
replace words with semantic neighbors.
"""

import random
from typing import List, Dict, Optional, Tuple
import gensim.downloader as api
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.corpus import stopwords
from tqdm.auto import tqdm


class KNEOAnonymizer:
    """Class for anonymization using KNEO (Knowledge-based NEighbor Operation)."""
    
    def __init__(
        self,
        embedding_model: str = "glove-wiki-gigaword-300",
        verbose: bool = True
    ):
        """
        Initialize the KNEO anonymizer.
        
        Args:
            embedding_model: Name of the embedding model to load
                Available options:
                - "glove-wiki-gigaword-50"
                - "glove-wiki-gigaword-100"
                - "glove-wiki-gigaword-200"
                - "glove-wiki-gigaword-300"
                - "fasttext-wiki-news-subwords-300"
            verbose: Show loading messages
        """
        self.stop_words = set(stopwords.words('english'))
        self.detokenizer = TreebankWordDetokenizer()
        self.neighbor_cache: Dict[str, List[str]] = {}
        self.verbose = verbose
        
        # Load the embedding model
        if self.verbose:
            print(f"Loading embedding model: {embedding_model}...")
        
        try:
            self.model = api.load(embedding_model)
            if self.verbose:
                print("✅ Model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.model = None
    
    def get_neighbors(
        self,
        word: str,
        k: int = 10,
        use_cache: bool = True
    ) -> List[str]:
        """
        Get the k semantic neighbors of a word.
        
        Args:
            word: Word to find neighbors for
            k: Number of neighbors to find
            use_cache: Use cache to speed up searches
            
        Returns:
            List of neighboring words (excluding the word itself)
        """
        if self.model is None:
            return []
        
        # Check the cache
        if use_cache and word in self.neighbor_cache:
            return self.neighbor_cache[word]
        
        word_lower = word.lower()
        
        try:
            neighbors = self.model.most_similar(positive=[word], topn=k)
            valid_neighbors = [
                n[0] for n in neighbors
                if n[0].lower() != word_lower
            ]
            
            if use_cache:
                self.neighbor_cache[word] = valid_neighbors
            
            return valid_neighbors
        
        except KeyError:
            # Word not in vocabulary
            if use_cache:
                self.neighbor_cache[word] = []
            return []
    
    def anonymize(
        self,
        sentence: str,
        k: int = 10,
        strategy: str = "random"
    ) -> str:
        """
        Anonymize a single sentence.
        
        Args:
            sentence: Sentence to anonymize
            k: Number of neighbors to consider
            strategy: Selection strategy ("random" or "first")
            
        Returns:
            Anonymized sentence
        """
        if self.model is None:
            raise ValueError("Embedding model not loaded")
        
        words = word_tokenize(sentence)
        if not words:
            return sentence
        
        new_words = list(words)
        
        # Find words to replace (in vocabulary and not stopwords)
        vocab_words_indices = [
            i for i, word in enumerate(words)
            if word in self.model and word.lower() not in self.stop_words
        ]
        
        if not vocab_words_indices:
            return sentence
        
        # Replace the words
        for i in vocab_words_indices:
            original_word = words[i]
            neighbors = self.get_neighbors(original_word, k=k)
            
            if neighbors:
                # Choose the neighbor
                if strategy == "first":
                    chosen_neighbor = neighbors[0]
                else:  # random
                    chosen_neighbor = random.choice(neighbors)
                
                # Clean the neighbor
                cleaned_neighbor = chosen_neighbor.replace("_", " ").replace("-", " ")
                if cleaned_neighbor:
                    new_words[i] = cleaned_neighbor
        
        # Rebuild the sentence
        return self.detokenizer.detokenize(new_words)
    
    def anonymize_batch(
        self,
        sentences: List[str],
        k: int = 10,
        strategy: str = "random",
        show_progress: bool = True
    ) -> List[str]:
        """
        Anonymize a batch of sentences.
        
        Args:
            sentences: List of sentences to anonymize
            k: Number of neighbors to consider
            strategy: Selection strategy ("random" or "first")
            show_progress: Show progress bar
            
        Returns:
            List of anonymized sentences
        """
        if self.model is None:
            raise ValueError("Embedding model not loaded")
        
        anonymized = []
        iterator = tqdm(sentences, desc="KNEO Anonymization") if show_progress else sentences
        
        for sentence in iterator:
            anonymized_sentence = self.anonymize(sentence, k=k, strategy=strategy)
            anonymized.append(anonymized_sentence)
        
        return anonymized
    
    def get_cache_stats(self) -> Dict[str, int]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {
            "cache_size": len(self.neighbor_cache),
            "total_neighbors": sum(len(v) for v in self.neighbor_cache.values())
        }
    
    def clear_cache(self):
        """Clear the neighbor cache."""
        self.neighbor_cache.clear()
