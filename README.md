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
- **Vendor A Rows Ingested**: 101,446
- **Vendor B Rows Ingested**: 18,400
- **Total Ingested Rows**: 119,846
- **Pandas Wall Time**: 25.6937 seconds
- **Pandas Peak Memory**: 27.54 MiB
- **Pandas Throughput**: 4,664.41 rows/sec
- **Polars Wall Time**: 0.2716 seconds
- **Polars Peak Memory**: 0.01 MiB
- **Polars Throughput**: 441,240.93 rows/sec
- **Polars/Pandas Speedup Ratio**: 94.6x

### Entity Resolution
- **Total Vendor A Entities**: 200
- **Matched Entities**: 183
- **Unmatched Entities**: 17
- **Match Rate**: 91.5% (0.915)
- **Resolution Precision**: 1.0
- **Resolution Recall**: 0.9946
- **Matched Residual Count**: 17 (Vendor A), 0 (Vendor B)

### Textual Features (Vendor C Corpus)
- **Document Count**: 1,600
- **Vocabulary Size**: 22
- **Mean Sentiment (Without Negation)**: 0.0
- **Mean Sentiment (With Negation)**: 0.0
- **Sign Flip Count**: 0

### Quality-Check Gate (`qc`)
| Check Name | Passed | Observed Value | Threshold |
|---|---|---|---|
| `check_schema` | True | `['entity_id', 'date', 'value_a', 'unit_scale_applied', 'entity_name_raw', 'value_b']` | Required columns present |
| `check_null_rate` | True | 0.7713 (77.13%) | <= 0.90 |
| `check_duplicate_keys` | True | 0 | 0 |
| `check_timestamp_monotonicity` | True | 0 violations | 0 |
| `check_value_ranges` | True | 0 negative values | 0 |
| `check_unit_consistency` | True | 0 inconsistencies | 0 |
| `check_distribution_drift` | True | 0.6686 (KS/Mean ratio) | < 10.0 |

### Discrepancy Classes
- `unit_scale_mismatch`: 2,024
- `revision_lag`: 1,450
- `missing_in_vendor_b`: 16
- `date_convention_skew`: 320
- `duplicate_key_divergence`: 1,446

### Test Coverage
- **Test Count**: 6 tests passing across 5 suites
- **Coverage**: 31% (`pytest -q --cov=src/vtn --cov-report=term-missing`)
- **Type check**: `python -m mypy src/vtn` -> Success: no issues found in 11 source files (pandas/polars stubs ignored, documented in `pyproject.toml`)

### Reproducibility
- Running `python -m vtn gen --seed 7` twice produces identical raw feeds and injection counts.
- Running `python -m vtn resolve` twice produces a byte-identical `data/out/resolution.json` (verified with `diff`).

---

## Findings & Trust Verdicts
See [FINDINGS.md](FINDINGS.md) for the complete breakdown of discrepancy classes and per-field trust verdicts (`value_a` preferred for timeliness/frequency; `value_b` preferred for CIK governance).

## Limitations
- Synthetic raw feeds generate stochastic variations.
- Lexicon-based sentiment scoring uses a static finance lexicon.
- Entity resolution threshold (0.8) may require fine-tuning for highly skewed entity naming variants.
- Reproducibility verified across multiple runs with identical seed.
