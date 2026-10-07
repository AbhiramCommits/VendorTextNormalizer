from vtn.text_features import (
    clean_text,
    tokenize,
    apply_negation,
    load_lexicon,
    extract_doc_features,
    compute_tfidf,
)


def test_negation_flips_sentiment_sign():
    lexicon = load_lexicon()
    text = "The company did not report growth and did not deliver growth."
    feats = extract_doc_features(text, lexicon)
    assert feats["negated_positive_count"] >= 2
    # Negating the only positive terms drives the score negative.
    assert feats["sentiment_without_negation"] > 0
    assert feats["sentiment_with_negation"] < 0


def test_apply_negation_marks_window_and_stops_at_clause():
    tokens = tokenize("we did not improve margins , but growth continued")
    marked = apply_negation(tokens)
    assert "not_improve" in marked
    assert "not_margins" in marked
    # "growth" is after a clause break, so it must not be negated.
    assert "not_growth" not in marked


def test_tfidf_hand_computed():
    docs = [["a", "b", "a"], ["b", "c"]]
    vectors, idf, counts = compute_tfidf(docs)
    # N=2; df(a)=1, df(b)=2, df(c)=1
    assert idf["b"] < idf["a"]
    assert abs(vectors[0]["a"] - (2 / 3) * idf["a"]) < 1e-12
    assert abs(vectors[0]["b"] - (1 / 3) * idf["b"]) < 1e-12
    assert counts == [3, 2]


def test_tokenizer_fallback_works():
    assert tokenize("Hello, World! 123") == ["hello", "world", "123"]
    # Smart quotes normalize to plain quotes and are then stripped by the cleaner;
    # the em dash normalizes to a hyphen which is retained.
    assert clean_text("\u201cSmart\u201d \u2014 dash") == "Smart - dash"
