# Vendor Data Discrepancy Findings Report

## Executive Summary
This report analyzes discrepancies between Vendor A (numeric time series) and Vendor B (SEC-style JSONL) ingested across 60835 missing entity segments and 66075 total classified discrepancy events.

## Discrepancy Classes & Counts
| Discrepancy Class | Count | Description |
|---|---|---|
| `unit_scale_mismatch` | 2024 | Scale differences (thousands vs millions) |
| `revision_lag` | 1450 | Timing lag in preliminary vs final revisions |
| `missing_in_vendor_b` | 60835 | Entities present in Vendor A but absent in Vendor B |
| `date_convention_skew` | 320 | Calendar alignment skew between reporting dates |
| `duplicate_key_divergence` | 1446 | Duplicate submission records |

## Per-Field Trust Verdicts
- **Vendor A (`value_a`)**: PREFERRED_TIMELINESS — Higher observation frequency and revision flags, though prone to unit scale shifts.
- **Vendor B (`value_b`)**: PREFERRED_GOVERNANCE — Strict CIK alignment and SEC-style governance, but suffers from missing entities (~8%) and lower frequency.

## Limitations
- Synthetic data generator introduces controlled stochastic drift.
- Lexicon-based sentiment analysis relies on Loughran-McDonald static subsets.
