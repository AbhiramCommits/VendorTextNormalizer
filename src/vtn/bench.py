import time
import tracemalloc
import json
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import polars as pl
from vtn.ingest_pandas import load_vendor_a, load_vendor_b, normalize_schema as normalize_pandas, assert_paths_agree
from vtn.ingest_polars import load_vendor_a_pl, load_vendor_b_pl, normalize_schema_pl

def run_benchmarks() -> Dict[str, Any]:
    """Runs performance benchmarks comparing pandas and polars ingest pipelines over raw vendor feeds.
    
    Measures wall-clock time (3 repeats, median), peak memory via tracemalloc, and compute throughput.
    """
    raw_dir = Path("data/raw")
    out_dir = Path("data/out")
    out_dir.mkdir(parents=True, exist_ok=True)

    a_path = raw_dir / "vendor_a.parquet"
    b_path = raw_dir / "vendor_b.jsonl"

    pd_times = []
    pd_mem_peaks = []
    pd_rows = 0
    for _ in range(3):
        tracemalloc.start()
        t0 = time.perf_counter()
        
        df_a_raw = load_vendor_a(a_path)
        df_b_raw = load_vendor_b(b_path)
        df_a_norm = normalize_pandas(df_a_raw, "vendor_a")
        df_b_norm = normalize_pandas(df_b_raw, "vendor_b")
        df_pd = pd.concat([df_a_norm, df_b_norm], ignore_index=True)
        
        t1 = time.perf_counter()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        pd_times.append(t1 - t0)
        pd_mem_peaks.append(peak / (1024 * 1024))
        pd_rows = len(df_pd)

    pd_median_time = sorted(pd_times)[1]
    pd_median_mem = sorted(pd_mem_peaks)[1]
    pd_rows_sec = pd_rows / pd_median_time

    pl_times = []
    pl_mem_peaks = []
    pl_rows = 0
    for _ in range(3):
        tracemalloc.start()
        t0 = time.perf_counter()
        
        df_a_raw_pl = load_vendor_a_pl(a_path)
        df_b_raw_pl = load_vendor_b_pl(b_path)
        df_a_norm_pl = normalize_schema_pl(df_a_raw_pl, "vendor_a")
        df_b_norm_pl = normalize_schema_pl(df_b_raw_pl, "vendor_b")
        df_pl = pl.concat([df_a_norm_pl, df_b_norm_pl])
        
        t1 = time.perf_counter()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        pl_times.append(t1 - t0)
        pl_mem_peaks.append(peak / (1024 * 1024))
        pl_rows = len(df_pl)

    pl_median_time = sorted(pl_times)[1]
    pl_median_mem = sorted(pl_mem_peaks)[1]
    pl_rows_sec_pl = pl_rows / pl_median_time

    assert_paths_agree(df_pd, df_pl.to_pandas())

    results: Dict[str, Any] = {
        "pandas": {
            "time_seconds": round(pd_median_time, 4),
            "peak_mib": round(pd_median_mem, 2),
            "rows_per_sec": round(pd_rows_sec, 2),
            "total_rows": pd_rows
        },
        "polars": {
            "time_seconds": round(pl_median_time, 4),
            "peak_mib": round(pl_median_mem, 2),
            "rows_per_sec": round(pl_rows_sec_pl, 2),
            "total_rows": pl_rows
        },
        "speedup_ratio": round(pd_median_time / pl_median_time, 2) if pl_median_time > 0 else 1.0
    }

    json_path = out_dir / "bench.json"
    json_path.write_text(json.dumps(results, indent=2))

    md_content = f"""# Ingest Performance Benchmark

| Engine | Wall Time (s) | Peak Memory (MiB) | Throughput (rows/sec) | Total Rows |
|---|---|---|---|---|
| **Pandas** | {results['pandas']['time_seconds']} | {results['pandas']['peak_mib']} | {results['pandas']['rows_per_sec']} | {results['pandas']['total_rows']} |
| **Polars** | {results['polars']['time_seconds']} | {results['polars']['peak_mib']} | {results['polars']['rows_per_sec']} | {results['polars']['total_rows']} |

- **Polars/Pandas Speedup Ratio**: {results['speedup_ratio']}x
"""
    md_path = out_dir / "bench.md"
    md_path.write_text(md_content)

    return results
