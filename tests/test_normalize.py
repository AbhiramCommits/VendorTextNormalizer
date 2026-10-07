import pandas as pd
from vtn.resolve import normalize_name
from vtn.ingest_pandas import parse_date_series

def test_normalize_name():
    assert normalize_name("ACME CORP.") == "acme"
    assert normalize_name("  Wayne Enterprises Inc.  ") == "wayne enterprises"
    assert normalize_name("Stark Industries LLC") == "stark industries"

def test_parse_date_series():
    s = pd.Series(["2022-01-03", "03/15/2022", "2022Q1"])
    parsed = parse_date_series(s)
    assert parsed.iloc[0] == "2022-01-03"
    assert parsed.iloc[1] == "2022-03-15"
    assert parsed.iloc[2] == "2022-03-31"
