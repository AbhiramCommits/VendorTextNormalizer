import json
from pathlib import Path
import math
import pandas as pd
import numpy as np
from vtn.ingest_pandas import parse_date_series


def _load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loads the panel plus raw vendor A and B feeds for measurement."""
    out_dir = Path("data/out")
    raw_dir = Path("data/raw")
    panel = pd.read_parquet(out_dir / "panel.parquet")
    raw_a = pd.read_csv(raw_dir / "vendor_a.csv")
    raw_b = pd.read_json(raw_dir / "vendor_b.jsonl", lines=True)
    return panel, raw_a, raw_b


def _unit_scale_checks(value_a: pd.Series, value_b: pd.Series) -> int:
    """Counts rows whose |log10 ratio| is consistent with a thousands/millions scale error."""
    both = pd.DataFrame({"a": value_a, "b": value_b}).dropna()
    both = both[(both["a"].abs() > 0) & (both["b"].abs() > 0)]
    if both.empty:
        return 0
    log_ratio = np.log10((both["a"].abs() / both["b"].abs()).values)
    near_3 = np.abs(np.abs(log_ratio) - 3.0) < 0.05
    near_6 = np.abs(np.abs(log_ratio) - 6.0) < 0.05
    return int(np.sum(near_3 | near_6))


def _sign_errors(value_a: pd.Series, value_b: pd.Series) -> int:
    """Counts rows where matched non-zero values have opposite signs."""
    both = pd.DataFrame({"a": value_a, "b": value_b}).dropna()
    both = both[(both["a"] != 0) & (both["b"] != 0)]
    return int(np.sum(np.sign(both["a"]) != np.sign(both["b"])))


def _stale_values(raw_a: pd.DataFrame) -> int:
    """Counts consecutive observations within a series that repeat the previous value."""
    df = raw_a[["series_id", "obs_date", "value"]].copy()
    df = df.sort_values(["series_id", "obs_date"])
    prev = df.groupby("series_id")["value"].shift(1)
    return int((df["value"] == prev).sum())


def _date_convention_skew(panel: pd.DataFrame) -> int:
    """Counts vendor B observations whose date is absent in vendor A but within +/-7 days of one.

    This measures near-miss calendar alignment between the two vendors, which is the
    observable footprint of differing date conventions (quarter-end vs daily).
    """
    skew = 0
    for _, grp in panel.groupby("entity_id"):
        a_dates = pd.to_datetime(grp.loc[grp["value_a"].notna(), "date"]).sort_values().values
        b_dates = pd.to_datetime(grp.loc[grp["value_b"].notna(), "date"]).sort_values().values
        if len(a_dates) == 0:
            continue
        a_i = a_dates.astype("datetime64[D]")
        for bd in b_dates:
            if bd in a_dates:
                continue
            if len(a_i) and np.min(np.abs((a_i - np.datetime64(bd, "D")).astype("timedelta64[D]").astype(int))) <= 7:
                skew += 1
    return int(skew)


def _revision_lag(raw_a: pd.DataFrame) -> int:
    """Counts observations still flagged PRELIMINARY in vendor A (subject to later revision)."""
    if "revision_flag" not in raw_a.columns:
        return 0
    return int((raw_a["revision_flag"].astype(str).str.upper() == "PRELIMINARY").sum())


def _duplicate_keys(raw_a: pd.DataFrame, raw_b: pd.DataFrame) -> int:
    """Counts genuinely duplicated records (identical across all columns) in both vendors."""
    da = int(raw_a.duplicated().sum())
    db = int(raw_b.duplicated().sum())
    return da + db


def _vendor_quality(raw_a: pd.DataFrame, raw_b: pd.DataFrame, panel: pd.DataFrame) -> dict:
    """Computes measurable quality metrics used to decide per-field trust.

    Dates are parsed to a single convention before monotonicity is checked, and
    duplicates are judged on identical records so date-format collapses are not
    mistaken for duplicate submissions.
    """
    a_null = float(raw_a["value"].isna().mean())
    b_null = float(raw_b["val"].isna().mean())
    a_dup = float(raw_a.duplicated().mean())
    b_dup = float(raw_b.duplicated().mean())

    def mono_violations(df: pd.DataFrame, key: str, date_col: str, cache: dict) -> int:
        if date_col in cache:
            parsed = cache[date_col]
        else:
            parsed = parse_date_series(df[date_col])
            cache[date_col] = parsed
        v = 0
        tmp = pd.DataFrame({key: df[key].values, "d": pd.to_datetime(parsed, errors="coerce").values})
        for _, grp in tmp.groupby(key):
            d = grp["d"].dropna()
            if len(d) > 1 and not d.is_monotonic_increasing:
                v += 1
        return v

    a_mono = mono_violations(raw_a, "series_id", "obs_date", {})
    b_mono = mono_violations(raw_b, "cik", "period", {})

    a_entities = raw_a["series_id"].nunique()
    b_entities = raw_b["cik"].nunique()
    b_coverage = b_entities / a_entities if a_entities else 0.0

    return {
        "value_a": {
            "null_rate": round(a_null, 6),
            "duplicate_rate": round(a_dup, 6),
            "monotonicity_violations": a_mono,
            "entity_count": int(a_entities),
        },
        "value_b": {
            "null_rate": round(b_null, 6),
            "duplicate_rate": round(b_dup, 6),
            "monotonicity_violations": b_mono,
            "entity_count": int(b_entities),
            "coverage_vs_vendor_a": round(b_coverage, 6),
        },
    }


def classify_discrepancies() -> dict:
    """Classifies field disagreements between the two numeric vendors into named classes.

    Every count is measured from the panel or the raw feeds; nothing is hardcoded.
    Writes ``data/out/discrepancies.json``.
    """
    out_dir = Path("data/out")
    panel, raw_a, raw_b = _load_inputs()

    counts = {
        "unit_scale_mismatch": _unit_scale_checks(panel["value_a"], panel["value_b"]),
        "sign_error": _sign_errors(panel["value_a"], panel["value_b"]),
        "stale_value": _stale_values(raw_a),
        "missing_in_vendor_b": int(panel["value_b"].isna().sum()),
        "date_convention_skew": _date_convention_skew(panel),
        "revision_lag": _revision_lag(raw_a),
        "duplicate_key_divergence": _duplicate_keys(raw_a, raw_b),
    }

    quality = _vendor_quality(raw_a, raw_b, panel)

    # Per-field trust: pick the vendor with the lower blended defect rate, with an
    # explicit, measurable reason. Defect rate blends null rate, duplicate rate and
    # monotonicity violations per entity.
    def defect_score(q: dict) -> float:
        return q["null_rate"] + q["duplicate_rate"] + q["monotonicity_violations"] / max(q["entity_count"], 1)

    a_score = defect_score(quality["value_a"])
    b_score = defect_score(quality["value_b"])
    b_covers = quality["value_b"].get("coverage_vs_vendor_a", 0.0)

    verdicts = {
        "value_a": {
            "trust_verdict": "PREFERRED" if a_score <= b_score else "SECONDARY",
            "reason": (
                f"blended defect score {a_score:.6f} (null_rate {quality['value_a']['null_rate']}, "
                f"dup_rate {quality['value_a']['duplicate_rate']}, mono_violations {quality['value_a']['monotonicity_violations']}); "
                f"daily frequency with explicit revision flags"
            ),
            "defect_score": round(a_score, 6),
        },
        "value_b": {
            "trust_verdict": "PREFERRED" if b_score < a_score else "SECONDARY",
            "reason": (
                f"blended defect score {b_score:.6f} (null_rate {quality['value_b']['null_rate']}, "
                f"dup_rate {quality['value_b']['duplicate_rate']}, mono_violations {quality['value_b']['monotonicity_violations']}); "
                f"entity coverage {b_covers:.4f} versus vendor A"
            ),
            "defect_score": round(b_score, 6),
        },
    }

    result = {
        "discrepancy_counts": counts,
        "vendor_quality": quality,
        "trust_verdicts": verdicts,
    }

    (out_dir / "discrepancies.json").write_text(json.dumps(result, indent=2))
    return result


def generate_findings_report():
    """Renders FINDINGS.md from the measured discrepancy JSON artifacts."""
    disc = classify_discrepancies()
    c = disc["discrepancy_counts"]

    md = f"""# Vendor Data Discrepancy Findings Report

## Executive Summary
This report compares Vendor A (numeric time series) and Vendor B (SEC-style JSONL) on the
normalized (entity_id, date) panel. All counts below are measured from `data/out/panel.parquet`
and the raw feeds in `data/raw/`; none are estimated.

## Discrepancy Classes & Counts
| Discrepancy Class | Count | How it is measured |
|---|---|---|
| `unit_scale_mismatch` | {c['unit_scale_mismatch']} | matched rows with \\|log10(a/b)\\| within 0.05 of 3 or 6 |
| `sign_error` | {c['sign_error']} | matched non-zero rows with opposite signs |
| `stale_value` | {c['stale_value']} | consecutive same-series observations with an unchanged value |
| `missing_in_vendor_b` | {c['missing_in_vendor_b']} | panel rows where value_b is null |
| `date_convention_skew` | {c['date_convention_skew']} | vendor B dates absent in A but within +/-7 days of an A date |
| `revision_lag` | {c['revision_lag']} | vendor A rows still flagged PRELIMINARY |
| `duplicate_key_divergence` | {c['duplicate_key_divergence']} | duplicated (key, date) rows across both numeric vendors |

## Per-Field Trust Verdicts
- **Vendor A (`value_a`)** — **{disc['trust_verdicts']['value_a']['trust_verdict']}**: {disc['trust_verdicts']['value_a']['reason']}
- **Vendor B (`value_b`)** — **{disc['trust_verdicts']['value_b']['trust_verdict']}**: {disc['trust_verdicts']['value_b']['reason']}

## Limitations
- Raw feeds are synthetically generated with seeded, deliberately injected defects.
- Lexicon-based sentiment analysis relies on a small static finance lexicon.
- The date-convention skew detector uses a +/-7 day window and is sensitive to that choice.
"""
    findings_path = Path("FINDINGS.md")
    findings_path.write_text(md)
    return findings_path
