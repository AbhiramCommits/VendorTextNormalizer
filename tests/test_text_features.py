from vtn.text_features import clean_text, tokenize, apply_negation, load_lexicon, extract_doc_features

def test_text_pipeline():
    text = "We did not improve operating margins, and observed no material weakness."
    cleaned = clean_text(text)
    tokens = tokenize(cleaned)
    neg_tokens = apply_negation(tokens)
    
    assert "not_improve" in neg_tokens
    assert "not_weakness" in neg_tokens
    
    lexicon = load_lexicon()
    feats = extract_doc_features(text, lexicon)
    assert feats['token_count'] > 0
    assert 'sentiment_score' in feats
