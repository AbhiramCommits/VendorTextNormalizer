import pandas as pd
from pathlib import Path
from datetime import datetime

def parse_date_series(s: pd.Series) -> pd.Series:
    """Parses ISO, US (MM/DD/YYYY), and Quarter (YYYYQn) strings into YYYY-MM-DD date strings."""
    def parse_one(val):
        if pd.isna(val):
            return pd.NaT
        val_str = str(val).strip()
        # Check ISO YYYY-MM-DD
        if len(val_str) == 10 and val_str[4] == '-' and val_str[7] == '-':
            try:
                return pd.to_datetime(val_str).strftime('%Y-%m-%d')
            except Exception:
                pass
        # Check US MM/DD/YYYY
        if '/' in val_str:
            try:
                return pd.to_datetime(val_str, format='%m/%d/%Y').strftime('%Y-%m-%d')
            except Exception:
                try:
                    return pd.to_datetime(val_str).strftime('%Y-%m-%d')
                except Exception:
                    pass
        # Check Quarter YYYYQn
        if 'Q' in val_str.upper():
            try:
                parts = val_str.upper().split('Q')
                year = int(parts[0])
                q = int(parts[1])
                month = q * 3
                # end of quarter month
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
            return pd.to_datetime(val_str).strftime('%Y-%m-%d')
        except Exception:
            return pd.NaT

    return s.apply(parse_one)

def load_vendor_a(path: Path) -> pd.DataFrame:
    """Loads Vendor A from parquet or CSV."""
    if str(path).endswith('.parquet'):
        return pd.read_parquet(path)
    else:
        return pd.read_csv(path)

def load_vendor_b(path: Path) -> pd.DataFrame:
    """Loads Vendor B from JSONL."""
    return pd.read_json(path, lines=True)

def load_vendor_c(path: Path) -> pd.DataFrame:
    """Loads Vendor C from JSONL."""
    return pd.read_json(path, lines=True)

def normalize_schema(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Lowercases columns, strips whitespace, parses dates, and outputs canonical schema:
       entity_key, entity_name_raw, date, value, unit, source
    """
    df = df.copy()
    # rename columns to lowercase / strip
    df.columns = [c.strip().lower() for c in df.columns]
    
    out = pd.DataFrame()
    if source == "vendor_a":
        out['entity_key'] = df['series_id'].astype(str).str.strip()
        out['entity_name_raw'] = df['company_name'].astype(str).str.strip()
        out['date'] = parse_date_series(df['obs_date'])
        out['value'] = pd.to_numeric(df['value'], errors='coerce')
        out['unit'] = df['unit'].astype(str).str.strip()
        out['source'] = 'vendor_a'
    elif source == "vendor_b":
        out['entity_key'] = df['cik'].astype(str).str.strip()
        out['entity_name_raw'] = df['issuer'].astype(str).str.strip()
        out['date'] = parse_date_series(df['period'])
        out['value'] = pd.to_numeric(df['val'], errors='coerce')
        out['unit'] = df['currency'].astype(str).str.strip()
        out['source'] = 'vendor_b'
    else:
        raise ValueError(f"Unknown source: {source}")
        
    return out

def assert_paths_agree(df_pd: pd.DataFrame, df_pl: pd.DataFrame):
    """Asserts pandas and polars canonical dataframes agree."""
    pd.testing.assert_frame_equal(
        df_pd.sort_values(by=['entity_key', 'date']).reset_index(drop=True),
        df_pl.sort_values(by=['entity_key', 'date']).reset_index(drop=True),
        check_like=True
    )
