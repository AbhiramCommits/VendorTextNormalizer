import re
import json
from pathlib import Path
import pandas as pd
import numpy as np
from collections import Counter

def clean_text(s: str) -> str:
    """Normalizes unicode, fixes smart quotes/em dashes, collapses whitespace."""
    if not isinstance(s, str):
        return ""
    # Smart quotes & em dashes
    s = s.replace("“", "\"").replace("”", "\"").replace("‘", "'").replace("’", "'")
    s = s.replace("—", "-").replace("–", "-")
    # Normalize unicode
    s = re.sub(r'[^\w\s\.\,\;\:\-\!\?]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def tokenize(s: str) -> list[str]:
    """Tokenizes text using regex (offline fallback)."""
    s_clean = clean_text(s)
    # Simple word tokenization
    return re.findall(r'\b\w+\b', s_clean.lower())

def apply_negation(tokens: list[str], window: int = 3) -> list[str]:
    """Marks tokens following a negation cue with NOT_ prefix within window."""
    negation_cues = {"not", "no", "never", "without", "failed", "did", "nor"}
    clause_puncts = {".", ",", ";", ":", "but", "however"}
    
    out = []
    i = 0
    n = len(tokens)
    while i < n:
        tok = tokens[i]
        if tok in negation_cues:
            out.append(tok)
            # Apply NOT_ prefix to next `window` tokens, stopping at clause punctuation
            w = 0
            j = i + 1
            while j < n and w < window:
                next_tok = tokens[j]
                if next_tok in clause_puncts:
                    break
                out.append(f"not_{next_tok}")
                j += 1
                w += 1
            i = j
        else:
            out.append(tok)
            i += 1
    return out

def load_lexicon() -> dict:
    lex_path = Path("src/vtn/lexicons/finance_sentiment.tsv")
    lexicon = {"positive": set(), "negative": set(), "uncertainty": set(), "litigious": set()}
    if lex_path.exists():
        for line in lex_path.read_text().splitlines():
            parts = line.split("\t")
            if len(parts) == 2:
                cat, word = parts[0].strip(), parts[1].strip()
                if cat in lexicon:
                    lexicon[cat].add(word)
    return lexicon

def extract_doc_features(text: str, lexicon: dict) -> dict:
    cleaned = clean_text(text)
    raw_tokens = tokenize(cleaned)
    neg_tokens = apply_negation(raw_tokens)
    
    token_count = len(raw_tokens)
    vocab = set(raw_tokens)
    ttr = len(vocab) / token_count if token_count > 0 else 0.0
    
    # Lexicon scoring
    pos_count = sum(1 for t in raw_tokens if t in lexicon["positive"])
    neg_count = sum(1 for t in raw_tokens if t in lexicon["negative"])
    unc_count = sum(1 for t in raw_tokens if t in lexicon["uncertainty"])
    
    negated_pos = sum(1 for t in neg_tokens if t.startswith("not_") and t[4:] in lexicon["positive"])
    negated_neg = sum(1 for t in neg_tokens if t.startswith("not_") and t[4:] in lexicon["negative"])
    
    # Adjusted sentiment score
    net_sentiment = (pos_count - neg_count - negated_pos + negated_neg) / (token_count + 1)
    uncertainty_density = unc_count / (token_count + 1)
    
    return {
        "token_count": token_count,
        "vocab_size": len(vocab),
        "ttr": round(ttr, 4),
        "positive_count": pos_count,
        "negative_count": neg_count,
        "negated_positive_count": negated_pos,
        "negated_negative_count": negated_neg,
        "uncertainty_density": round(uncertainty_density, 4),
        "sentiment_score": round(net_sentiment, 6),
        "tokens": raw_tokens,
        "neg_tokens": neg_tokens
    }

def attach_to_panel() -> dict:
    out_dir = Path("data/out")
    panel_path = out_dir / "panel.parquet"
    vendor_c_path = Path("data/raw/vendor_c.jsonl")
    
    if not panel_path.exists() or not vendor_c_path.exists():
        raise FileNotFoundError("Panel or Vendor C data missing. Run gen and resolve first.")
        
    panel = pd.read_parquet(panel_path)
    df_c = pd.read_json(vendor_c_path, lines=True)
    
    lexicon = load_lexicon()
    
    # Extract features for each document in Vendor C
    doc_results = []
    sentiment_without_neg_list = []
    sentiment_with_neg_list = []
    sign_flips = 0
    
    for _, row in df_c.iterrows():
        text = row['body']
        cleaned = clean_text(text)
        raw_tokens = tokenize(cleaned)
        neg_tokens = apply_negation(raw_tokens)
        
        token_count = len(raw_tokens)
        pos_count = sum(1 for t in raw_tokens if t in lexicon["positive"])
        neg_count = sum(1 for t in raw_tokens if t in lexicon["negative"])
        
        # Without negation
        sent_no_neg = (pos_count - neg_count) / (token_count + 1)
        
        # With negation
        negated_pos = sum(1 for t in neg_tokens if t.startswith("not_") and t[4:] in lexicon["positive"])
        negated_neg = sum(1 for t in neg_tokens if t.startswith("not_") and t[4:] in lexicon["negative"])
        sent_with_neg = (pos_count - neg_count - negated_pos + negated_neg) / (token_count + 1)
        
        sentiment_without_neg_list.append(sent_no_neg)
        sentiment_with_neg_list.append(sent_with_neg)
        
        if np.sign(sent_no_neg) != np.sign(sent_with_neg) and sent_no_neg != 0 and sent_with_neg != 0:
            sign_flips += 1
            
        feats = extract_doc_features(text, lexicon)
        feats['entity_id'] = row['cik'] # mapped via cik
        feats['date'] = row['filed_date']
        doc_results.append(feats)

    doc_df = pd.DataFrame(doc_results)
    
    # Aggregate to entity_id and date (quarter-to-day forward fill across panel dates)
    # Ensure doc_df date is datetime
    doc_df['date'] = pd.to_datetime(doc_df['date'])
    panel['date'] = pd.to_datetime(panel['date'])
    
    # Sort for merge / asof or left join on exact date or quarter forward fill
    # Since panel has daily obs and Vendor C has quarter-end dates, we forward-fill quarterly features to daily panel per entity
    doc_agg = doc_df.groupby(['entity_id', 'date'], as_index=False).agg({
        'sentiment_score': 'mean',
        'uncertainty_density': 'mean',
        'ttr': 'mean',
        'token_count': 'sum'
    })
    
    merged = pd.merge(panel, doc_agg, on=['entity_id', 'date'], how='left')
    # Forward fill text features within each entity
    text_cols = ['sentiment_score', 'uncertainty_density', 'ttr', 'token_count']
    merged[text_cols] = merged.groupby('entity_id')[text_cols].ffill()
    
    merged.to_parquet(out_dir / "panel_features.parquet", index=False)
    
    summary = {
        "doc_count": len(df_c),
        "vocab_size": int(doc_df['vocab_size'].mean()),
        "mean_sentiment_without_negation": float(np.mean(sentiment_without_neg_list)),
        "mean_sentiment_with_negation": float(np.mean(sentiment_with_neg_list)),
        "sign_flip_count": sign_flips
    }
    
    (out_dir / "text_features.json").write_text(json.dumps(summary, indent=2))
    return summary
