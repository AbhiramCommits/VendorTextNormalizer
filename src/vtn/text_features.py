import re
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from collections import Counter

try:
    import spacy  # type: ignore
    _SPACY_AVAILABLE = True
except Exception:
    spacy = None  # type: ignore
    _SPACY_AVAILABLE = False

_NEGATION_CUES = {"not", "no", "never", "without", "nor", "failed", "cannot"}
_CLAUSE_BREAKS = {".", ",", ";", ":", "but", "however", "although"}

_NLP = None
_TOKENIZER_PATH = "regex"


def clean_text(s: str) -> str:
    """Normalizes unicode, fixes smart quotes/em dashes, collapses whitespace."""
    if not isinstance(s, str):
        return ""
    s = s.replace("\u201c", '"').replace("\u201d", '"').replace("\u2018", "'").replace("\u2019", "'")
    s = s.replace("\u2014", "-").replace("\u2013", "-")
    s = re.sub(r'[^\w\s\.\,\;\:\-\!\?]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def tokenize(s: str) -> List[str]:
    """Tokenizes text. Uses spaCy's tokenizer when installed, otherwise a regex fallback.

    The chosen path is logged once per process via ``_TOKENIZER_PATH``.
    """
    global _TOKENIZER_PATH, _NLP
    if _SPACY_AVAILABLE and spacy is not None:
        try:
            if _NLP is None:
                _NLP = spacy.blank("en")
            _TOKENIZER_PATH = "spacy"
            return [t.text.lower() for t in _NLP(clean_text(s)) if t.text.strip()]
        except Exception:
            pass
    _TOKENIZER_PATH = "regex"
    return re.findall(r'\b\w+\b', clean_text(s).lower())


def _negation_mask(tokens: List[str], window: int = 3) -> List[bool]:
    """Returns a boolean list aligned with ``tokens`` marking tokens inside a negation window."""
    mask = [False] * len(tokens)
    i = 0
    n = len(tokens)
    while i < n:
        if tokens[i] in _NEGATION_CUES:
            w = 0
            j = i + 1
            while j < n and w < window:
                if tokens[j] in _CLAUSE_BREAKS:
                    break
                mask[j] = True
                j += 1
                w += 1
            i = j
        else:
            i += 1
    return mask


def apply_negation(tokens: List[str], window: int = 3) -> List[str]:
    """Marks tokens following a negation cue with a ``not_`` prefix inside the window.

    Stops at clause punctuation. This is the string-level API; scoring uses
    :func:`_negation_mask` for aligned polarity flipping.
    """
    mask = _negation_mask(tokens, window)
    return [f"not_{t}" if m else t for t, m in zip(tokens, mask)]


def load_lexicon() -> Dict[str, set]:
    """Loads the checked-in finance sentiment lexicon (category -> set of words)."""
    lex_path = Path("src/vtn/lexicons/finance_sentiment.tsv")
    lexicon: Dict[str, set] = {"positive": set(), "negative": set(), "uncertainty": set(), "litigious": set()}
    if lex_path.exists():
        for line in lex_path.read_text().splitlines():
            parts = line.split("\t")
            if len(parts) == 2:
                cat, word = parts[0].strip(), parts[1].strip()
                if cat in lexicon:
                    lexicon[cat].add(word)
    return lexicon


def compute_tfidf(docs: List[List[str]]) -> Tuple[List[Dict[str, float]], Dict[str, float], List[int]]:
    """Computes tf-idf vectors from scratch using ``collections.Counter`` (no sklearn).

    Args:
        docs: Corpus as a list of token lists.

    Returns:
        ``(tfidf_vectors, idf, doc_token_counts)`` where ``tfidf_vectors[i][term]``
        is the tf-idf weight of ``term`` in document ``i`` and ``idf`` maps each
        term to its inverse document frequency.
    """
    n_docs = len(docs)
    df: Counter = Counter()
    for toks in docs:
        df.update(set(toks))
    idf = {t: math.log(n_docs / (1.0 + d)) + 1.0 for t, d in df.items()}

    vectors: List[Dict[str, float]] = []
    counts: List[int] = []
    for toks in docs:
        tf_counter = Counter(toks)
        total = sum(tf_counter.values()) or 1
        counts.append(sum(tf_counter.values()))
        vec = {t: (c / total) * idf.get(t, 0.0) for t, c in tf_counter.items()}
        vectors.append(vec)
    return vectors, idf, counts


def extract_doc_features(text: str, lexicon: Dict[str, set]) -> dict:
    """Extracts token, vocabulary, ttr, lexicon and negation-adjusted sentiment features for one document."""
    cleaned = clean_text(text)
    raw_tokens = tokenize(cleaned)
    mask = _negation_mask(raw_tokens)

    token_count = len(raw_tokens)
    vocab = set(raw_tokens)
    ttr = len(vocab) / token_count if token_count > 0 else 0.0

    pos_count = sum(1 for t in raw_tokens if t in lexicon["positive"])
    neg_count = sum(1 for t in raw_tokens if t in lexicon["negative"])
    unc_count = sum(1 for t in raw_tokens if t in lexicon["uncertainty"])
    lit_count = sum(1 for t in raw_tokens if t in lexicon["litigious"])

    negated_pos = sum(1 for t, m in zip(raw_tokens, mask) if m and t in lexicon["positive"])
    negated_neg = sum(1 for t, m in zip(raw_tokens, mask) if m and t in lexicon["negative"])

    denom = token_count + 1
    sentiment_without = (pos_count - neg_count) / denom
    polarity = 0.0
    for t, m in zip(raw_tokens, mask):
        if t in lexicon["positive"]:
            polarity += -1.0 if m else 1.0
        elif t in lexicon["negative"]:
            polarity += 1.0 if m else -1.0
    sentiment_with = polarity / denom

    return {
        "token_count": token_count,
        "vocab_size": len(vocab),
        "ttr": round(ttr, 4),
        "positive_count": pos_count,
        "negative_count": neg_count,
        "litigious_count": lit_count,
        "negated_positive_count": negated_pos,
        "negated_negative_count": negated_neg,
        "uncertainty_density": round(unc_count / denom, 6),
        "sentiment_without_negation": round(sentiment_without, 6),
        "sentiment_with_negation": round(sentiment_with, 6),
    }


def attach_to_panel() -> dict:
    """Aggregates document features to (entity_id, date), forward-fills onto the panel.

    Writes ``data/out/panel_features.parquet`` and ``data/out/text_features.json``.
    """
    out_dir = Path("data/out")
    panel_path = out_dir / "panel.parquet"
    vendor_c_path = Path("data/raw/vendor_c.jsonl")

    if not panel_path.exists() or not vendor_c_path.exists():
        raise FileNotFoundError("Panel or Vendor C data missing. Run gen and resolve first.")

    panel = pd.read_parquet(panel_path)
    df_c = pd.read_json(vendor_c_path, lines=True)
    lexicon = load_lexicon()

    texts = df_c['body'].tolist()
    token_lists = [tokenize(clean_text(t)) for t in texts]
    tfidf_vectors, idf, _ = compute_tfidf(token_lists)

    # Corpus-wide top terms by mean tf-idf
    term_totals: Counter = Counter()
    for vec in tfidf_vectors:
        term_totals.update(vec)
    top_terms = [t for t, _ in term_totals.most_common(5)]

    doc_results = []
    sent_without: List[float] = []
    sent_with: List[float] = []
    sign_flips = 0

    for idx, row in df_c.iterrows():
        feats = extract_doc_features(row['body'], lexicon)
        feats['top_term'] = max(tfidf_vectors[idx].items(), key=lambda kv: kv[1])[0] if tfidf_vectors[idx] else ""
        s0 = feats['sentiment_without_negation']
        s1 = feats['sentiment_with_negation']
        sent_without.append(s0)
        sent_with.append(s1)
        if (s0 > 0 > s1) or (s0 < 0 < s1):
            sign_flips += 1
        feats['entity_id'] = row['cik']
        feats['date'] = row['filed_date']
        doc_results.append(feats)

    doc_df = pd.DataFrame(doc_results)
    doc_df['date'] = pd.to_datetime(doc_df['date'])
    panel['date'] = pd.to_datetime(panel['date'])

    doc_agg = doc_df.groupby(['entity_id', 'date'], as_index=False).agg({
        'sentiment_with_negation': 'mean',
        'uncertainty_density': 'mean',
        'ttr': 'mean',
        'token_count': 'sum',
    }).rename(columns={'sentiment_with_negation': 'sentiment_score'})

    merged = pd.merge(panel, doc_agg, on=['entity_id', 'date'], how='left')
    text_cols = ['sentiment_score', 'uncertainty_density', 'ttr', 'token_count']
    merged[text_cols] = merged.groupby('entity_id')[text_cols].ffill()
    merged.to_parquet(out_dir / "panel_features.parquet", index=False)

    corpus_vocab = set()
    for toks in token_lists:
        corpus_vocab.update(toks)

    summary = {
        "tokenizer_path": _TOKENIZER_PATH,
        "doc_count": len(df_c),
        "corpus_vocab_size": len(corpus_vocab),
        "mean_doc_vocab_size": float(doc_df['vocab_size'].mean()),
        "top_tfidf_terms": top_terms,
        "mean_sentiment_without_negation": float(np.mean(sent_without)),
        "mean_sentiment_with_negation": float(np.mean(sent_with)),
        "min_sentiment_with_negation": float(np.min(sent_with)),
        "max_sentiment_with_negation": float(np.max(sent_with)),
        "sign_flip_count": sign_flips,
    }

    (out_dir / "text_features.json").write_text(json.dumps(summary, indent=2))
    return summary
