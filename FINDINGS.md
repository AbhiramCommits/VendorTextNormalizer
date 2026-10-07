# Vendor Data Discrepancy Findings Report

## Executive Summary
This report compares Vendor A (numeric time series) and Vendor B (SEC-style JSONL) on the
normalized (entity_id, date) panel. All counts below are measured from `data/out/panel.parquet`
and the raw feeds in `data/raw/`; none are estimated.

## Discrepancy Classes & Counts
| Discrepancy Class | Count | How it is measured |
|---|---|---|
| `unit_scale_mismatch` | 5849 | matched rows with \|log10(a/b)\| within 0.05 of 3 or 6 |
| `sign_error` | 0 | matched non-zero rows with opposite signs |
| `stale_value` | 1424 | consecutive same-series observations with an unchanged value |
| `missing_in_vendor_b` | 60909 | panel rows where value_b is null |
| `date_convention_skew` | 3570 | vendor B dates absent in A but within +/-7 days of an A date |
| `revision_lag` | 2025 | vendor A rows still flagged PRELIMINARY |
| `duplicate_key_divergence` | 1661 | duplicated (key, date) rows across both numeric vendors |

## Per-Field Trust Verdicts
- **Vendor A (`value_a`)** — **SECONDARY**: blended defect score 1.046358 (null_rate 0.029989, dup_rate 0.016369, mono_violations 200); daily frequency with explicit revision flags
- **Vendor B (`value_b`)** — **PREFERRED**: blended defect score 0.019783 (null_rate 0.019783, dup_rate 0.0, mono_violations 0); entity coverage 0.9200 versus vendor A

## Limitations
- Raw feeds are synthetically generated with seeded, deliberately injected defects.
- Lexicon-based sentiment analysis relies on a small static finance lexicon.
- The date-convention skew detector uses a +/-7 day window and is sensitive to that choice.
