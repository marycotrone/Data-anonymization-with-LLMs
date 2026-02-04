"""
Easy Data Augmentation (EDA) for text anonymization.

This module implements the four EDA techniques:
- Synonym Replacement (SR): Replace words with synonyms
- Random Insertion (RI): Randomly insert synonyms
- Random Swap (RS): Randomly swap words
- Random Deletion (RD): Randomly delete words
"""

import random
from typing import List, Set
from nltk.corpus import wordnet, stopwords
from tqdm.auto import tqdm


class EDAAnonymizer:
    """Class for anonymization using Easy Data Augmentation."""
    
    # Punctuation marks not to swap
    PUNCTUATION_MARKS: Set[str] = {'!', '?', '"', '.', ',', ';', '(', ')', ':'}
    
    def __init__(self, seed: int = 42):
        """
        Initialize the EDA anonymizer.
        
        Args:
            seed: Seed for reproducibility
        """
        random.seed(seed)
        self.stop_words = set(stopwords.words('english'))
    
    def get_synonyms(self, word: str) -> List[str]:
        """
        Get synonyms of a word from WordNet.
        
        Args:
            word: Word to find synonyms for
            
        Returns:
            List of synonyms (excluding the word itself)
        """
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace("_", " ").replace("-", " ").lower()
                synonyms.add(synonym)
        
        if word in synonyms:
            synonyms.remove(word)
        
        return list(synonyms)
    
    def synonym_replacement(self, words: List[str], alpha: float) -> List[str]:
        """
        Replace N random words with synonyms.
        
        Args:
            words: List of words
            alpha: Percentage of words to replace
            
        Returns:
            List of words with replacements applied
        """
        new_words = words.copy()
        num_words_to_replace = round(len(words) * alpha)
        
        # Find replaceable words (not stopwords and alphabetic)
        replaceable = [
            i for i, word in enumerate(words)
            if word.lower() not in self.stop_words and word.isalpha()
        ]
        
        n = min(num_words_to_replace, len(replaceable))
        indices_to_replace = random.sample(replaceable, n)
        
        for i in indices_to_replace:
            word = new_words[i]
            synonyms = self.get_synonyms(word)
            if synonyms:
                new_words[i] = random.choice(synonyms)
        
        return new_words
    
    def random_deletion(self, words: List[str], alpha: float) -> List[str]:
        """
        Delete random words with probability alpha.
        
        Args:
            words: List of words
            alpha: Deletion probability for each word
            
        Returns:
            List of words with deletions applied
        """
        if len(words) <= 1:
            return words
        
        new_words = []
        for word in words:
            if word.isalnum():
                if random.uniform(0, 1) > alpha:
                    new_words.append(word)
            else:
                new_words.append(word)
        
        # Ensure at least one alphanumeric word survived
        if not any(w.isalnum() for w in new_words):
            original_alphanums = [w for w in words if w.isalnum()]
            if original_alphanums:
                return [random.choice(original_alphanums)]
        
        return new_words
    
    def random_swap(self, words: List[str], alpha: float) -> List[str]:
        """
        Swap random word pairs (excluding punctuation).
        
        Args:
            words: List of words
            alpha: Percentage of swaps to perform
            
        Returns:
            List of words with swaps applied
        """
        new_words = words.copy()
        
        # Find swappable indices (not punctuation)
        swappable_indices = [
            i for i, word in enumerate(new_words)
            if word not in self.PUNCTUATION_MARKS
        ]
        
        num_words = len(swappable_indices)
        if num_words < 2:
            return new_words
        
        num_swaps = round(num_words * alpha)
        
        for _ in range(num_swaps):
            idx1, idx2 = random.sample(swappable_indices, 2)
            new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
        
        return new_words
    
    def random_insertion(self, words: List[str], alpha: float) -> List[str]:
        """
        Insert random synonyms at random positions.
        
        Args:
            words: List of words
            alpha: Percentage of insertions to perform
            
        Returns:
            List of words with insertions applied
        """
        new_words = words.copy()
        num_words_to_insert = round(len(words) * alpha)
        
        # Find words to look for synonyms
        possible_insertion_syn = [
            word for word in words
            if word.lower() not in self.stop_words and word.isalpha()
        ]
        
        if not possible_insertion_syn:
            return new_words
        
        for _ in range(num_words_to_insert):
            random_word = random.choice(possible_insertion_syn)
            synonyms = self.get_synonyms(random_word)
            
            if synonyms:
                random_synonym = random.choice(synonyms)
                random_index = random.randint(0, len(new_words) - 1)
                new_words.insert(random_index, random_synonym)
        
        return new_words
    
    def anonymize(
        self,
        sentence: str,
        alpha_sr: float = 0.5,
        alpha_ri: float = 0.5,
        alpha_rs: float = 0.5,
        alpha_rd: float = 0.5
    ) -> str:
        """
        Apply all EDA techniques in sequence.
        
        Args:
            sentence: Sentence to anonymize
            alpha_sr: Alpha for Synonym Replacement
            alpha_ri: Alpha for Random Insertion
            alpha_rs: Alpha for Random Swap
            alpha_rd: Alpha for Random Deletion
            
        Returns:
            Anonymized sentence
        """
        words = sentence.split()
        words = [word for word in words if word]
        
        if not words:
            return ""
        
        # Apply techniques in sequence
        words = self.synonym_replacement(words, alpha_sr)
        words = self.random_insertion(words, alpha_ri)
        words = self.random_swap(words, alpha_rs)
        words = self.random_deletion(words, alpha_rd)
        
        return ' '.join(words)
    
    def anonymize_batch(
        self,
        sentences: List[str],
        alpha_sr: float = 0.5,
        alpha_ri: float = 0.5,
        alpha_rs: float = 0.5,
        alpha_rd: float = 0.5,
        show_progress: bool = True
    ) -> List[str]:
        """
        Anonymize a batch of sentences.
        
        Args:
            sentences: List of sentences to anonymize
            alpha_sr: Alpha for Synonym Replacement
            alpha_ri: Alpha for Random Insertion
            alpha_rs: Alpha for Random Swap
            alpha_rd: Alpha for Random Deletion
            show_progress: Show progress bar
            
        Returns:
            List of anonymized sentences
        """
        anonymized = []
        iterator = tqdm(sentences, desc="EDA Anonymization") if show_progress else sentences
        
        for sentence in iterator:
            anonymized_sentence = self.anonymize(
                sentence, alpha_sr, alpha_ri, alpha_rs, alpha_rd
            )
            anonymized.append(anonymized_sentence)
        
        return anonymized
