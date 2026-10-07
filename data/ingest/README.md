# Ingest & Vendor Feed Documentation

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
