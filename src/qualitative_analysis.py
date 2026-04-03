import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_qual_data(original_texts, all_anonymized):
    """Build a flat DataFrame with original and all anonymized texts."""
    rows = [{'text': t, 'dataset_source': 'Original'} for t in original_texts]
    for name, anonymized in all_anonymized.items():
        rows.extend({'text': t, 'dataset_source': name} for t in anonymized)
    df = pd.DataFrame(rows)
    print(f"Dataset prepared: {len(df)} total samples "
          f"({len(original_texts)} original + {len(all_anonymized)} configurations)")
    return df


# ---------------------------------------------------------------------------
# Step 1 – Readability
# ---------------------------------------------------------------------------

def calculate_readability(df_qual):
    """Add Flesch Reading Ease and Flesch-Kincaid Grade Level columns."""
    try:
        import textstat
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'textstat', '-q'])
        import textstat

    from tqdm import tqdm
    tqdm.pandas()

    def _calc(text):
        if not isinstance(text, str) or not text.strip():
            return np.nan, np.nan
        return textstat.flesch_reading_ease(text), textstat.flesch_kincaid_grade(text)

    print("Calculating readability metrics...")
    df_qual[['flesch_reading_ease', 'flesch_kincaid_grade']] = (
        df_qual['text'].progress_apply(lambda x: pd.Series(_calc(x)))
    )
    print("Readability metrics calculated.")
    return df_qual


# ---------------------------------------------------------------------------
# Step 2 – Fluency (Perplexity via GPT-2)
# ---------------------------------------------------------------------------

def calculate_perplexity(df_qual, batch_size=8):
    """Add perplexity and log_perplexity columns using GPT-2."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from tqdm import tqdm

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    model.eval()
    tokenizer.pad_token = tokenizer.eos_token

    texts = df_qual['text'].tolist()
    ppl_values = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Calculating Perplexity"):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True,
                        truncation=True, max_length=512)
        input_ids     = enc.input_ids.to(device)
        attention_mask = enc.attention_mask.to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=input_ids)

        logits        = outputs.logits
        shift_logits  = logits[:, :-1, :].contiguous()
        shift_labels  = input_ids[:, 1:].contiguous()
        shift_mask    = attention_mask[:, 1:].contiguous()

        loss_fct     = torch.nn.CrossEntropyLoss(reduction="none")
        token_losses = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        ).view(shift_labels.size())

        sentence_loss = (token_losses * shift_mask).sum(dim=1) / shift_mask.sum(dim=1)
        ppl_values.extend(torch.exp(sentence_loss).cpu().numpy())

    df_qual['perplexity']     = ppl_values
    df_qual['log_perplexity'] = np.log(df_qual['perplexity'])
    print("Perplexity calculated.")
    return df_qual


# ---------------------------------------------------------------------------
# Step 3 – Visualization helpers
# ---------------------------------------------------------------------------

def _get_clean_data(df, col, limit_range=None):
    data = df.dropna(subset=[col])
    data = data[~data[col].isin([np.inf, -np.inf])]
    if limit_range:
        data = data[(data[col] >= limit_range[0]) & (data[col] <= limit_range[1])]
    return data


def _get_custom_palette(df):
    """Per-group color palette: EDA=Reds, KNEO=Blues, GEMMA=Greens, LLAMA=Purples."""
    labels = df['dataset_source'].unique().tolist()

    eda_labels   = sorted([l for l in labels if l.upper().startswith('EDA')])
    kneo_labels  = sorted([l for l in labels if l.upper().startswith('KNEO')])
    gemma_labels = sorted([l for l in labels if l.upper().startswith('GEMMA')])
    llama_labels = sorted([l for l in labels if l.upper().startswith('LLAMA')])

    def _pal(name, n, offset=2):
        return sns.color_palette(name, n_colors=n + offset)[offset:]

    palette = {'Original': 'hotpink'}
    for i, l in enumerate(eda_labels):   palette[l] = _pal('Reds',    len(eda_labels))[i]
    for i, l in enumerate(kneo_labels):  palette[l] = _pal('Blues',   len(kneo_labels))[i]
    for i, l in enumerate(gemma_labels): palette[l] = _pal('Greens',  len(gemma_labels))[i]
    for i, l in enumerate(llama_labels): palette[l] = _pal('Purples', len(llama_labels))[i]
    for l in labels:
        if l not in palette:
            palette[l] = 'gray'
    return palette


def plot_violin_comparison(df, col, title, x_limit=None, save_path=None):
    """Horizontal violin plot for distribution comparison."""
    n = len(df['dataset_source'].unique())
    plt.figure(figsize=(12, 0.6 * n + 2))

    clean_df = _get_clean_data(df, col, x_limit)
    palette  = _get_custom_palette(clean_df)
    order    = clean_df.groupby("dataset_source")[col].median().sort_values().index

    sns.violinplot(
        data=clean_df, x=col, y="dataset_source", hue="dataset_source",
        order=order, orient="h", linewidth=1, inner="box",
        palette=palette, legend=False, cut=0
    )
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel(col, fontsize=12)
    plt.ylabel("Dataset Source", fontsize=12)
    if x_limit:
        plt.xlim(*x_limit)
    plt.grid(axis='x', alpha=0.5)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()


def plot_ridge_comparison(df, col, title, x_limit=None, save_path=None):
    """Ridge (joy) plot for distribution comparison."""
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

    clean_df = _get_clean_data(df, col, x_limit)
    palette  = _get_custom_palette(clean_df)

    g = sns.FacetGrid(clean_df, row="dataset_source", hue="dataset_source",
                      aspect=12, height=0.7, palette=palette)
    g.map(sns.kdeplot, col, clip_on=False, fill=True, alpha=0.7, lw=1.5, bw_adjust=.6)
    g.map(sns.kdeplot, col, clip_on=False, color="w", lw=2, bw_adjust=.6)
    g.map(plt.axhline, y=0, lw=2, clip_on=False)

    def _label(*_, label, **__):
        ax = plt.gca()
        ax.text(0, .2, label, fontweight="bold", color="black",
                ha="left", va="center", transform=ax.transAxes)

    g.map(_label, col)
    g.figure.subplots_adjust(hspace=-0.5)
    g.set_titles("")
    g.set(yticks=[])
    g.despine(bottom=True, left=True)
    plt.suptitle(title, y=0.98, fontsize=16, fontweight='bold')
    if x_limit:
        plt.xlim(*x_limit)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()
    sns.set_theme(style="whitegrid")


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_qualitative_analysis(original_texts, all_anonymized,
                              results_dir="../results", batch_size=8):
    """
    Run the full qualitative analysis pipeline:
    prepare → readability → perplexity → plots.
    Returns the enriched df_qual DataFrame.
    """
    df_qual = prepare_qual_data(original_texts, all_anonymized)
    df_qual = calculate_readability(df_qual)
    df_qual = calculate_perplexity(df_qual, batch_size=batch_size)

    print("\nGenerating plots...")
    plot_ridge_comparison(df_qual, 'flesch_reading_ease',
                          '1. Readability - Flesch Reading Ease',
                          x_limit=(-10, 110),
                          save_path=f"{results_dir}/ridge_readability.png")
    plot_ridge_comparison(df_qual, 'flesch_kincaid_grade',
                          '2. Complexity - Flesch-Kincaid Grade Level',
                          x_limit=(0, 40),
                          save_path=f"{results_dir}/ridge_complexity.png")
    plot_ridge_comparison(df_qual, 'log_perplexity',
                          '3. Fluency - Log Perplexity (GPT-2)',
                          x_limit=(0, 10),
                          save_path=f"{results_dir}/ridge_fluency.png")

    print("\nQualitative analysis completed!")
    return df_qual
