import pandas as pd
from vtn.ingest_pandas import normalize_schema as norm_pd
from vtn.ingest_polars import normalize_schema_pl as norm_pl
import polars as pl

def test_engine_parity():
    raw_a = pd.DataFrame({
        "series_id": ["SER_001"],
        "company_name": ["Acme"],
        "obs_date": ["2022-01-01"],
        "value": [10.0],
        "unit": ["USD"]
    })
    
    pd_norm = norm_pd(raw_a, "vendor_a")
    pl_norm = norm_pl(pl.DataFrame(raw_a), "vendor_a").to_pandas()
    
    pd.testing.assert_frame_equal(
        pd_norm.sort_index(axis=1),
        pl_norm.sort_index(axis=1)
    )
