# VendorTextNormalizer

VendorTextNormalizer ingests two messy public vendor feeds covering the same entities—one numeric time series (Vendor A), one free-text corpus (Vendor B/C)—normalizes them onto a single `(entity_id, date)` key, extracts numeric and textual features with negation handling, runs a hard quality-check gate, and writes a structured findings report.

## What it does
1. **Generates & Ingests**: Produces realistic messy feeds with seeded data-quality defects (name variants, whitespace noise, date format skews, unit mismatches, duplicate keys, nulls, and drift).
2. **Normalizes & Resolves**: Harmonizes schemas and resolves entities across disparate vendor keys (Series ID vs CIK) using multi-stage matching.
3. **Extracts Features & QC**: Computes NLP features (tf-idf, sentiment, negation handling) and executes strict data quality gates.

## Architecture
```
gen -> ingest (pandas | polars) -> resolve -> features -> qc -> report
```
- **`gen`**: `src/vtn/gen.py` -> `data/raw/` (`vendor_a.csv`, `vendor_a.parquet`, `vendor_b.jsonl`, `vendor_c.jsonl`)
- **`ingest`**: `src/vtn/ingest_pandas.py`, `src/vtn/ingest_polars.py` -> `data/out/bench.json`, `bench.md`
- **`resolve`**: `src/vtn/resolve.py` -> `data/out/crosswalk.json`, `resolution.json`, `panel.parquet`
- **`features`**: `src/vtn/text_features.py` -> `data/out/panel_features.parquet`, `text_features.json`
- **`qc`**: `src/vtn/qc.py` -> `data/out/qc_report.json`
- **`report`**: `src/vtn/discrepancy.py` -> `FINDINGS.md`

### Canonical Panel Schema
- `entity_id`: String (normalized CIK / entity key)
- `entity_name_raw`: String (raw vendor entity name)
- `date`: Date (YYYY-MM-DD)
- `value_a`: Float (Vendor A harmonized numeric value)
- `value_b`: Float (Vendor B harmonized numeric value)
- `unit_scale_applied`: String (harmonized scale unit)
- `sentiment_score`: Float (textual sentiment score)
- `uncertainty_density`: Float (uncertainty word density)

---

## How to Run
```bash
make setup
python -m vtn all --seed 7
make test
make bench
python -m vtn qc
```

*Note on Real-World Data Swap*: To swap synthetic data for real feeds (e.g., FRED/EIA numeric series or SEC EDGAR filings), replace the files in `data/raw/` matching the schema specifications in `data/ingest/README.md`.

---

## Measured Results (Seed 7)

### Ingest & Throughput (Vendor A + Vendor B)
- **Vendor A Rows Ingested**: 101,472
- **Vendor B Rows Ingested**: 18,400
- **Total Ingested Rows**: 119,872
- **Pandas Wall Time**: 25.3937 seconds
- **Pandas Peak Memory**: 27.53 MiB
- **Pandas Throughput**: 4,720.53 rows/sec
- **Polars Wall Time**: 0.2739 seconds
- **Polars Peak Memory**: 0.01 MiB
- **Polars Throughput**: 437,639.52 rows/sec
- **Polars/Pandas Speedup Ratio**: 92.71x

### Entity Resolution
- **Total Vendor A Entities**: 200
- **Matched Entities**: 183
- **Unmatched Entities**: 107 (Vendor A names without a B counterpart)
- **Match Rate**: 91.5% (0.915)
- **Methods**: `normalized_exact` 161, `exact_name` 22
- **Resolution Precision**: 1.0
- **Resolution Recall**: 0.9946
- **Unmatched Residual**: 107 (Vendor A), 1 (Vendor B)

### Textual Features (Vendor C Corpus)
- **Document Count**: 1,600
- **Corpus Vocabulary Size**: 341
- **Mean Document Vocabulary**: 80.88 tokens
- **Top tf-idf Terms**: `the`, `and`, `company`, `in`, `we`
- **Mean Sentiment (Without Negation)**: +0.024358
- **Mean Sentiment (With Negation)**: -0.017282
- **Sentiment Range (With Negation)**: [-0.125786, +0.100629]
- **Sign-Flip Count**: 590 of 1,600 documents flip sign when negation is applied
- **Tokenizer Path**: `regex` (spaCy not installed)

### Quality-Check Gate (`qc`)
All checks measured on `data/out/panel.parquet`. `check_distribution_drift` is on the
documented expected-fail list because the generator injects a scale shift in the last 10%
of dates by design; the gate passes as long as no *unexpected* check fails.

| Check Name | Passed | Observed Value | Threshold |
|---|---|---|---|
| `check_schema` | True | required columns present | required columns present |
| `check_null_rate` | True | 0.7715 | <= 0.90 |
| `check_duplicate_keys` | True | 0 | 0 |
| `check_timestamp_monotonicity` | True | 0 violations | 0 |
| `check_value_ranges` | True | 0 negative values | 0 |
| `check_unit_consistency` | True | 0 inconsistencies | 0 |
| `check_distribution_drift` | False (expected) | 0.3280 (two-sample KS) | < 0.25 |

Corruption demo: injecting 10 duplicate keys plus 5 negative values makes three checks fail
(`check_duplicate_keys`=10, `check_timestamp_monotonicity`=1, `check_value_ranges`=5) and
`python -m vtn qc` exits **1**; the clean run exits **0**.

### Discrepancy Classes
| Class | Count |
|---|---|
| `unit_scale_mismatch` | 5,849 |
| `sign_error` | 0 |
| `stale_value` | 1,424 |
| `missing_in_vendor_b` | 60,909 |
| `date_convention_skew` | 3,570 |
| `revision_lag` | 2,025 |
| `duplicate_key_divergence` | 1,661 |

### Test Coverage
- **Test Count**: 9 tests passing across 5 suites
- **Coverage**: 31% (`pytest -q --cov=src/vtn --cov-report=term-missing`)
- **Type check**: `python -m mypy src/vtn` -> Success: no issues found in 11 source files (pandas/polars stubs ignored, documented in `pyproject.toml`)

### Reproducibility
- Running `python -m vtn gen --seed 7` twice produces identical raw feeds and injection counts.
- Running `python -m vtn resolve` twice produces a byte-identical `data/out/resolution.json` (verified with `diff`).

---

## Findings & Trust Verdicts
See [FINDINGS.md](FINDINGS.md) for the full table. Measured per-field verdicts:
- **Vendor B (`value_b`)** — **PREFERRED**: blended defect score 0.019783 (null rate 0.019783, duplicate rate 0.0, 0 monotonicity violations), entity coverage 0.92 versus Vendor A.
- **Vendor A (`value_a`)** — **SECONDARY**: blended defect score 1.046358 (null rate 0.029989, duplicate rate 0.016369, 200 apparent monotonicity violations caused by quarter-coarsened timestamps), but higher frequency with explicit revision flags.

## Limitations
- Raw feeds are synthetically generated with seeded, deliberately injected defects, not real downloads.
- Lexicon-based sentiment uses a small static finance word list; scores are lexicon-bound, not model-based.
- The 0.8 token-set Jaccard resolution threshold is sensitive: lowering it raises recall but risks false merges.
- The date-convention skew detector depends on a +/-7 day window choice.
- `sign_error` is 0 because all generated values are positive; the classifier supports negative values but the seed data does not exercise it.
