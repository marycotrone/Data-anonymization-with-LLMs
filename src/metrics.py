"""
Evaluation metrics for text anonymization.

Metrics:
- Irreversibility: Levenshtein Ratio, Jaccard Similarity
- Utility: Cosine Similarity (SBERT)
- Anonymization: NER Score
"""

import re
import numpy as np
from typing import List, Dict, Optional
from nltk.tokenize import word_tokenize
from nltk.metrics.distance import edit_distance
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
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
        self.verbose = verbose

        if self.verbose:
            print(f"Loading SBERT: {sbert_model}...")
        self.sbert = SentenceTransformer(sbert_model)

        if self.verbose:
            print(f"Loading Spacy: {spacy_model}...")
        self.nlp = spacy.load(spacy_model)

        if self.verbose:
            print("Models loaded")

    @staticmethod
    def clean_text_for_ner(text: str) -> str:
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def calculate_levenshtein_ratio(
        self,
        original_sentences: List[str],
        generated_sentences: List[str]
    ) -> float:
        scores = self.calculate_levenshtein_ratio_list(original_sentences, generated_sentences)
        return np.mean(scores) if scores else 0

    def calculate_levenshtein_ratio_list(
        self,
        original_sentences: List[str],
        generated_sentences: List[str]
    ) -> List[float]:
        scores = []
        for orig, gen in zip(original_sentences, generated_sentences):
            orig_tokens = word_tokenize(orig.lower())
            gen_tokens  = word_tokenize(gen.lower())
            if not orig_tokens and not gen_tokens:
                scores.append(1.0); continue
            if not orig_tokens or not gen_tokens:
                scores.append(0.0); continue
            distance = edit_distance(orig_tokens, gen_tokens)
            max_len  = max(len(orig_tokens), len(gen_tokens))
            scores.append(1.0 - distance / max_len)
        return scores

    def calculate_jaccard_similarity(
        self,
        original_sentences: List[str],
        generated_sentences: List[str]
    ) -> float:
        scores = self.calculate_jaccard_similarity_list(original_sentences, generated_sentences)
        return np.mean(scores) if scores else 0

    def calculate_jaccard_similarity_list(
        self,
        original_sentences: List[str],
        generated_sentences: List[str]
    ) -> List[float]:
        scores = []
        for orig, gen in zip(original_sentences, generated_sentences):
            orig_set = set(word_tokenize(orig.lower()))
            gen_set  = set(word_tokenize(gen.lower()))
            if not orig_set and not gen_set:
                scores.append(1.0); continue
            if not orig_set or not gen_set:
                scores.append(0.0); continue
            intersection = len(orig_set & gen_set)
            union        = len(orig_set | gen_set)
            scores.append(intersection / union)
        return scores

    def calculate_cosine_similarity(
        self,
        original_sentences: List[str],
        generated_sentences: List[str],
        show_progress: bool = True
    ) -> float:
        sims = self.calculate_cosine_similarity_list(original_sentences, generated_sentences, show_progress)
        return np.mean(sims) if len(sims) > 0 else 0

    def calculate_cosine_similarity_list(
        self,
        original_sentences: List[str],
        generated_sentences: List[str],
        show_progress: bool = True
    ) -> np.ndarray:
        if len(original_sentences) != len(generated_sentences):
            raise ValueError("Lists must have the same length")
        orig_emb = self.sbert.encode(original_sentences, show_progress_bar=show_progress)
        gen_emb  = self.sbert.encode(generated_sentences, show_progress_bar=show_progress)
        return np.diag(cosine_similarity(orig_emb, gen_emb))

    def calculate_ner_score(
        self,
        original_sentences: List[str],
        generated_sentences: List[str],
        strict_mode: bool = True,
        target_labels: List[str] = None
    ) -> float:
        scores = self.calculate_ner_score_list(original_sentences, generated_sentences, strict_mode, target_labels)
        return np.mean(scores) if scores else 1.0

    def calculate_ner_score_list(
        self,
        original_sentences: List[str],
        generated_sentences: List[str],
        strict_mode: bool = True,
        target_labels: List[str] = None
    ) -> List[float]:
        if target_labels is None:
            target_labels = ["PERSON", "GPE", "ORG", "LOC"]

        scores = []
        orig_cleaned = [self.clean_text_for_ner(s) for s in original_sentences]
        gen_cleaned  = [self.clean_text_for_ner(s) for s in generated_sentences]

        with self.nlp.select_pipes(disable=["parser", "tagger", "lemmatizer"]):
            for orig, gen in zip(orig_cleaned, gen_cleaned):
                doc_orig  = self.nlp(orig)
                orig_ents = [e for e in doc_orig.ents if e.label_ in target_labels]
                if not orig_ents:
                    continue
                gen_lower = gen.lower()
                leaked = 0
                for ent in orig_ents:
                    entity = ent.text.lower()
                    if strict_mode:
                        if any(t in gen_lower for t in entity.split() if len(t) > 2):
                            leaked += 1
                    else:
                        if entity in gen_lower:
                            leaked += 1
                scores.append(1.0 - leaked / len(orig_ents))
        return scores

    def evaluate_all(
        self,
        original_sentences: List[str],
        generated_sentences: List[str],
        show_progress: bool = True
    ) -> dict:
        results = {
            "levenshtein_ratio": self.calculate_levenshtein_ratio(original_sentences, generated_sentences),
            "jaccard_similarity": self.calculate_jaccard_similarity(original_sentences, generated_sentences),
            "cosine_similarity":  self.calculate_cosine_similarity(original_sentences, generated_sentences, show_progress),
            "ner_score":          self.calculate_ner_score(original_sentences, generated_sentences),
        }
        return results

    def plot_metric_distributions(
        self,
        scores_dict: Dict[str, List[float]],
        metric_name: str,
        save_path: Optional[str] = None
    ) -> None:
        """
        Ridge plot of metric score distributions.

        Labels are color-coded by method group:
          EDA   → Reds
          KNEO  → Blues
          GEMMA → Greens
          LLAMA → Purples
          other → gray

        Args:
            scores_dict: {label: [scores]} — keys should start with EDA/KNEO/GEMMA/LLAMA
            metric_name: Title label for the plot
            save_path: If given, save to this path; otherwise auto-save as <metric_name>.png
        """
        all_data = []
        labels_ordered = list(scores_dict.keys())

        for label, scores in scores_dict.items():
            if hasattr(scores, 'tolist'):
                scores = scores.tolist()
            for score in scores:
                all_data.append({"Score": score, "Dataset": label})

        if not all_data:
            print(f"No data to plot for {metric_name}")
            return

        df_plot = pd.DataFrame(all_data)

        # Group labels by method
        eda_labels   = [l for l in labels_ordered if l.upper().startswith('EDA')]
        kneo_labels  = [l for l in labels_ordered if l.upper().startswith('KNEO')]
        gemma_labels = [l for l in labels_ordered if 'GEMMA' in l.upper()]
        llama_labels = [l for l in labels_ordered if 'LLAMA' in l.upper()]

        def _palette(name, n, offset=2):
            return sns.color_palette(name, n_colors=n + offset)[offset:]

        custom_palette = {}
        for i, l in enumerate(eda_labels):
            custom_palette[l] = _palette("Reds",    len(eda_labels))[i]
        for i, l in enumerate(kneo_labels):
            custom_palette[l] = _palette("Blues",   len(kneo_labels))[i]
        for i, l in enumerate(gemma_labels):
            custom_palette[l] = _palette("Greens",  len(gemma_labels))[i]
        for i, l in enumerate(llama_labels):
            custom_palette[l] = _palette("Purples", len(llama_labels))[i]
        for l in labels_ordered:
            if l not in custom_palette:
                custom_palette[l] = "gray"

        sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

        g = sns.FacetGrid(
            df_plot, row="Dataset", hue="Dataset",
            aspect=9, height=1.0, palette=custom_palette
        )
        g.map(sns.kdeplot, "Score", bw_adjust=0.6, clip_on=False, fill=True, alpha=0.8, linewidth=0)
        g.map(sns.kdeplot, "Score", bw_adjust=0.6, clip_on=False, color="w", lw=2)
        g.map(plt.axhline, y=0, lw=1, clip_on=False, color="grey", alpha=0.2)

        def _label(*_, label, **__):
            ax = plt.gca()
            ax.text(0, 0.1, label, fontweight="bold", color="black",
                    ha="left", va="center", transform=ax.transAxes)

        g.map(_label, "Score")
        g.set_titles("")
        g.set(yticks=[], ylabel="")
        g.despine(bottom=True, left=True)

        plt.subplots_adjust(hspace=-0.8)
        g.figure.suptitle(f"Ridge Plot: {metric_name}", fontsize=18, fontweight='bold', y=0.98)
        plt.xlabel("Score", fontsize=14)

        out = save_path or f"{metric_name.replace(' ', '_')}.png"
        plt.savefig(out, dpi=300, bbox_inches='tight')
        if self.verbose:
            print(f"Plot saved: {out}")

        plt.show()
