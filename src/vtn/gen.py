import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List

def generate(seed: int, n_entities: int = 200, n_days: int = 500, out_dir: Path = Path("data/raw")) -> Dict[str, Any]:
    """Generates synthetic messy vendor feeds (Vendor A CSV/Parquet, Vendor B JSONL, Vendor C JSONL) and injection truth."""
    random.seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    start_date = datetime(2022, 1, 3)
    # generate business days roughly
    current = start_date
    dates = []
    for _ in range(n_days):
        while current.weekday() >= 5: # skip weekends
            current += timedelta(days=1)
        dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)

    # Entity generation
    base_names = [
        "ACME CORP", "GLOBEX INDUSTRIES", "STARK TECH", "WAYNE ENTERPRISES",
        "CYBERDYNE SYSTEMS", "UMBRELLA CORP", "MASSIVE DYNAMIC", "INITECH",
        "HUESOS INC", "SOYLENT CORP", "PIPERNET", "HOOLI", "VIRTUCON",
        "BLUTH COMPANY", "OSCORP", "STARK INDUSTRIES", "LEXCORP", "ABSTERGO",
        "TYRELL CORP", "WEYLAND YUTANI"
    ]
    
    entities = []
    for i in range(n_entities):
        series_id = f"SER_{i:04d}"
        cik = f"CIK{i:09d}"
        
        # Pick base name or generate
        if i < len(base_names):
            base = base_names[i]
        else:
            base = f"COMPANY {i} CORP"
            
        # Name variants for Vendor A & B
        variants_a = [
            base,
            base.replace("CORP", "Corporation").replace("INC", "Incorporated"),
            base + " INC.",
            base.title()
        ]
        name_a = random.choice(variants_a)
        
        variants_b = [
            base.lower().title(),
            base + " LLC",
            base.replace("CORP", "Company"),
            base
        ]
        name_b = random.choice(variants_b)
        
        entities.append({
            "entity_id": f"ENT_{i:04d}",
            "series_id": series_id,
            "cik": cik,
            "name_a": name_a,
            "name_b": name_b,
            "base_name": base
        })

    # Vendor B missing ~8% entities entirely
    missing_b_count = int(n_entities * 0.08)
    missing_b_indices = set(random.sample(range(n_entities), missing_b_count))

    vendor_a_rows = []
    vendor_b_rows = []
    vendor_c_rows = []

    injected_counts = {
        "name_variants": 0,
        "whitespace_noise": 0,
        "date_formats": {"ISO": 0, "US": 0, "QUARTER": 0},
        "unit_mismatches": 0,
        "duplicate_keys": 0,
        "nulls": 0,
        "non_monotonic_timestamps": 0,
        "scale_shift_drift": 0,
        "unicode_artifacts": 0,
        "missing_in_vendor_b": missing_b_count
    }

    date_format_choices = ["ISO", "US", "QUARTER"]

    for idx, ent in enumerate(entities):
        # Vendor A: series_id, company_name, obs_date, value, unit, revision_flag
        base_val = random.uniform(10.0, 1000.0)
        
        # introduce scale shift drift in last 10% dates for some entities
        drift_start_idx = int(n_days * 0.9)
        
        for d_idx, d_str in enumerate(dates):
            val = base_val * (1.0 + random.normalvariate(0, 0.05))
            if d_idx >= drift_start_idx and idx % 3 == 0:
                val *= 1000.0 # scale shift drift
                injected_counts["scale_shift_drift"] += 1

            unit = "USD_THOUSANDS"
            if random.random() < 0.02:
                unit = "USD_MILLIONS" # unit mismatch
                injected_counts["unit_mismatches"] += 1
                
            rev_flag = "FINAL" if d_idx < n_days - 10 else "PRELIMINARY"
            
            # Date formatting noise
            dt_obj = datetime.strptime(d_str, "%Y-%m-%d")
            fmt_choice = random.choice(date_format_choices)
            injected_counts["date_formats"][fmt_choice] += 1
            
            if fmt_choice == "ISO":
                obs_date_a = d_str
            elif fmt_choice == "US":
                obs_date_a = dt_obj.strftime("%m/%d/%Y")
            else:
                q = (dt_obj.month - 1) // 3 + 1
                obs_date_a = f"{dt_obj.year}Q{q}"

            name_a_noisy = ent["name_a"]
            if random.random() < 0.1:
                name_a_noisy = f"  {name_a_noisy} \t"
                injected_counts["whitespace_noise"] += 1
            if random.random() < 0.05:
                name_a_noisy = name_a_noisy.replace("CORP", "CORP.")
                injected_counts["name_variants"] += 1

            # Nulls injection (~2-4%)
            val_a = val
            if random.random() < 0.03:
                val_a = None
                injected_counts["nulls"] += 1

            row_a = {
                "series_id": ent["series_id"],
                "company_name": name_a_noisy,
                "obs_date": obs_date_a,
                "value": val_a,
                "unit": unit,
                "revision_flag": rev_flag
            }
            vendor_a_rows.append(row_a)
            
            # Duplicate keys (~1-2%)
            if random.random() < 0.015:
                vendor_a_rows.append(row_a)
                injected_counts["duplicate_keys"] += 1

        # Vendor B: cik, issuer, period, val, currency
        if idx not in missing_b_indices:
            for d_idx, d_str in enumerate(dates):
                # Sample weekly or monthly for Vendor B to differ slightly or daily
                if d_idx % 5 != 0: # subset to reduce size
                    continue
                val_b = base_val * (1.0 + random.normalvariate(0, 0.05))
                if random.random() < 0.02:
                    val_b = None
                    injected_counts["nulls"] += 1
                
                currency = "USD"
                name_b_noisy = ent["name_b"]
                if random.random() < 0.1:
                    name_b_noisy = f"\n{name_b_noisy}\r "
                    injected_counts["whitespace_noise"] += 1

                vendor_b_rows.append({
                    "cik": ent["cik"],
                    "issuer": name_b_noisy,
                    "period": d_str,
                    "val": val_b,
                    "currency": currency
                })

        # Vendor C (free text, JSONL): one per entity-quarter
        # Quarters across 2022-2023
        quarters = [("2022-03-31", "Q1 2022"), ("2022-06-30", "Q2 2022"), ("2022-09-30", "Q3 2022"), ("2022-12-31", "Q4 2022"),
                    ("2023-03-31", "Q1 2023"), ("2023-06-30", "Q2 2023"), ("2023-09-30", "Q3 2023"), ("2023-12-31", "Q4 2023")]
        
        for q_date, q_label in quarters:
            text_bodies = [
                f"During {q_label}, {ent['base_name']} did not improve operating margins, and we observed no material weakness in internal controls, although management remains uncertain about future growth.",
                f"{ent['base_name']} achieved solid revenue growth this quarter. However, supply chain bottlenecks failed to resolve completely, leading to litigation risks.",
                f"We did not experience significant headwinds. The company reported robust performance without any material weakness. Operations proceeded smoothly.",
                f"Management noted that previous forecasts were overly optimistic. Results failed to meet expectations, creating uncertainty and potential legal exposure."
            ]
            body = random.choice(text_bodies)
            
            # Unicode / em-dash / smart-quote artifacts
            if random.random() < 0.3:
                body = body.replace("", "“").replace("", "”").replace("--", "—")
                injected_counts["unicode_artifacts"] += 1

            vendor_c_rows.append({
                "cik": ent["cik"],
                "company": ent["name_b"],
                "filed_date": q_date,
                "section": "Item 7. Management's Discussion and Analysis",
                "body": body
            })

    # Write files
    import pandas as pd
    df_a = pd.DataFrame(vendor_a_rows)
    csv_path = out_dir / "vendor_a.csv"
    parquet_path = out_dir / "vendor_a.parquet"
    df_a.to_csv(csv_path, index=False)
    df_a.to_parquet(parquet_path, index=False)

    jsonl_b_path = out_dir / "vendor_b.jsonl"
    with open(jsonl_b_path, "w") as f:
        for row in vendor_b_rows:
            f.write(json.dumps(row) + "\n")

    jsonl_c_path = out_dir / "vendor_c.jsonl"
    with open(jsonl_c_path, "w") as f:
        for row in vendor_c_rows:
            f.write(json.dumps(row) + "\n")

    truth_path = out_dir / "_injected_truth.json"
    with open(truth_path, "w") as f:
        json.dump(injected_counts, f, indent=2)

    ingest_readme = out_dir.parent / "ingest" / "README.md"
    ingest_readme.parent.mkdir(parents=True, exist_ok=True)
    ingest_readme.write_text("""# Ingest & Vendor Feed Documentation

## Real-World Analogues
- **Vendor A (Numeric Series)**: Mirrors FRED (Federal Reserve Economic Data) or EIA time-series CSV/Parquet bulk feeds, containing series identifiers, entity names, observation dates, values, reporting units, and revision flags.
- **Vendor B (Numeric JSONL)**: Mirrors SEC EDGAR financial statement data sets or API JSON feeds, keyed by CIK, issuer name, reporting period, numeric values, and currency.
- **Vendor C (Free-Text JSONL)**: Mirrors SEC EDGAR 10-K/10-Q filing full-text documents containing CIK, company name, filing date, section headers (e.g., MD&A), and narrative body text.

## How to Swap for Real Downloads
To replace synthetic generated data with real feeds:
1. Download bulk CSV/Parquet files from FRED/EIA and place them matching `vendor_a.parquet`.
2. Download SEC EDGAR company facts JSONL / submission files matching `vendor_b.jsonl`.
3. Download parsed EDGAR filing sections matching `vendor_c.jsonl`.
4. Ensure schema column mappings in `src/vtn/ingest_pandas.py` and `src/vtn/ingest_polars.py` align with your downloaded schema columns.
""")

    return {
        "vendor_a_rows": len(df_a),
        "vendor_b_rows": len(vendor_b_rows),
        "vendor_c_rows": len(vendor_c_rows),
        "injected_counts": injected_counts
    }
