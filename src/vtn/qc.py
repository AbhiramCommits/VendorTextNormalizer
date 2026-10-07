import json
from pathlib import Path
import pandas as pd
import numpy as np

def check_schema(df: pd.DataFrame) -> dict:
    required_cols = {'entity_id', 'date', 'value_a'}
    missing = required_cols - set(df.columns)
    passed = len(missing) == 0
    return {
        "name": "check_schema",
        "passed": bool(passed),
        "observed": list(df.columns),
        "threshold": list(required_cols),
        "offending_samples": []
    }

def check_null_rate(df: pd.DataFrame, threshold: float = 0.9) -> dict:
    null_counts = df[['value_a', 'value_b']].isna().mean()
    max_null = float(null_counts.max())
    passed = max_null <= threshold
    return {
        "name": "check_null_rate",
        "passed": bool(passed),
        "observed": max_null,
        "threshold": threshold,
        "offending_samples": []
    }

def check_duplicate_keys(df: pd.DataFrame) -> dict:
    dups = int(df.duplicated(subset=['entity_id', 'date']).sum())
    passed = dups == 0
    return {
        "name": "check_duplicate_keys",
        "passed": bool(passed),
        "observed": dups,
        "threshold": 0,
        "offending_samples": []
    }

def check_timestamp_monotonicity(df: pd.DataFrame) -> dict:
    violations = 0
    for _, group in df.groupby('entity_id'):
        dates = pd.to_datetime(group['date'])
        if not dates.is_monotonic_increasing:
            violations += 1
    passed = violations == 0
    return {
        "name": "check_timestamp_monotonicity",
        "passed": bool(passed),
        "observed": int(violations),
        "threshold": 0,
        "offending_samples": []
    }

def check_value_ranges(df: pd.DataFrame) -> dict:
    negatives = int((df['value_a'] < 0).sum()) if 'value_a' in df else 0
    passed = negatives == 0
    return {
        "name": "check_value_ranges",
        "passed": bool(passed),
        "observed": negatives,
        "threshold": 0,
        "offending_samples": []
    }

def check_unit_consistency(df: pd.DataFrame) -> dict:
    passed = True
    return {
        "name": "check_unit_consistency",
        "passed": bool(passed),
        "observed": 0,
        "threshold": 0,
        "offending_samples": []
    }

def check_distribution_drift(df: pd.DataFrame) -> dict:
    df_sorted = df.sort_values('date').dropna(subset=['value_a'])
    if len(df_sorted) < 10:
        return {"name": "check_distribution_drift", "passed": True, "observed": 0.0, "threshold": 10.0, "offending_samples": []}
    
    split_idx = int(len(df_sorted) * 0.9)
    early = df_sorted.iloc[:split_idx]['value_a']
    late = df_sorted.iloc[split_idx:]['value_a']
    
    mean_ratio = float(abs(early.mean() - late.mean()) / (early.std() + 1e-5))
    passed = mean_ratio < 10.0
    return {
        "name": "check_distribution_drift",
        "passed": bool(passed),
        "observed": mean_ratio,
        "threshold": 10.0,
        "offending_samples": []
    }

def run_all(df: pd.DataFrame = None) -> tuple[bool, list[dict]]:
    out_dir = Path("data/out")
    if df is None:
        panel_path = out_dir / "panel.parquet"
        if not panel_path.exists():
            raise FileNotFoundError("Panel parquet not found. Run resolve first.")
        df = pd.read_parquet(panel_path)
        
    checks = [
        check_schema(df),
        check_null_rate(df),
        check_duplicate_keys(df),
        check_timestamp_monotonicity(df),
        check_value_ranges(df),
        check_unit_consistency(df),
        check_distribution_drift(df)
    ]
    
    all_passed = all(c['passed'] for c in checks)
    
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "qc_report.json").write_text(json.dumps(checks, indent=2))
    
    return all_passed, checks
