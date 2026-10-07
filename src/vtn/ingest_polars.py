import polars as pl
from pathlib import Path

def load_vendor_a_pl(path: Path) -> pl.DataFrame:
    """Loads Vendor A numeric feed (Parquet or CSV) into a Polars DataFrame."""
    if str(path).endswith('.parquet'):
        return pl.read_parquet(path)
    else:
        return pl.read_csv(path)

def load_vendor_b_pl(path: Path) -> pl.DataFrame:
    """Loads Vendor B numeric feed (JSONL) into a Polars DataFrame."""
    return pl.read_ndjson(path)

def load_vendor_c_pl(path: Path) -> pl.DataFrame:
    """Loads Vendor C free-text feed (JSONL) into a Polars DataFrame."""
    return pl.read_ndjson(path)

def parse_date_polars(col: pl.Expr) -> pl.Expr:
    """Parses a Polars string column containing ISO, US or YYYYQn dates to YYYY-MM-DD."""
    return col.map_elements(lambda x: _parse_one_date(x), return_dtype=pl.Utf8)

def _parse_one_date(val: object) -> str | None:
    """Parses a single date value in ISO, US (MM/DD/YYYY) or YYYYQn format to YYYY-MM-DD."""
    if val is None:
        return None
    val_str = str(val).strip()
    if len(val_str) == 10 and val_str[4] == '-' and val_str[7] == '-':
        return val_str
    if '/' in val_str:
        try:
            parts = val_str.split('/')
            if len(parts) == 3:
                m, d, y = parts
                return f"{y}-{m.zfill(2)}-{d.zfill(2)}"
        except Exception:
            pass
    if 'Q' in val_str.upper():
        try:
            parts = val_str.upper().split('Q')
            year = int(parts[0])
            q = int(parts[1])
            month = q * 3
            if month == 3:
                return f"{year}-03-31"
            elif month == 6:
                return f"{year}-06-30"
            elif month == 9:
                return f"{year}-09-30"
            elif month == 12:
                return f"{year}-12-31"
        except Exception:
            pass
    try:
        from datetime import datetime
        return datetime.fromisoformat(val_str).strftime('%Y-%m-%d')
    except Exception:
        return None

def normalize_schema_pl(df: pl.DataFrame, source: str) -> pl.DataFrame:
    """Normalizes a vendor frame (polars) to the canonical schema.

    Args:
        df: Raw vendor dataframe.
        source: Either ``"vendor_a"`` or ``"vendor_b"``.

    Returns:
        DataFrame with columns entity_key, entity_name_raw, date, value, unit, source.
    """
    df = df.rename({c: c.strip().lower() for c in df.columns})
    
    if source == "vendor_a":
        return df.select([
            pl.col("series_id").cast(pl.Utf8).str.strip_chars().alias("entity_key"),
            pl.col("company_name").cast(pl.Utf8).str.strip_chars().alias("entity_name_raw"),
            parse_date_polars(pl.col("obs_date")).alias("date"),
            pl.col("value").cast(pl.Float64, strict=False).alias("value"),
            pl.col("unit").cast(pl.Utf8).str.strip_chars().alias("unit"),
            pl.lit("vendor_a").alias("source")
        ])
    elif source == "vendor_b":
        return df.select([
            pl.col("cik").cast(pl.Utf8).str.strip_chars().alias("entity_key"),
            pl.col("issuer").cast(pl.Utf8).str.strip_chars().alias("entity_name_raw"),
            parse_date_polars(pl.col("period")).alias("date"),
            pl.col("val").cast(pl.Float64, strict=False).alias("value"),
            pl.col("currency").cast(pl.Utf8).str.strip_chars().alias("unit"),
            pl.lit("vendor_b").alias("source")
        ])
    else:
        raise ValueError(f"Unknown source: {source}")
