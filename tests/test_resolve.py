import pandas as pd
from vtn.resolve import build_crosswalk, resolution_report

def test_resolve_crosswalk():
    df_a = pd.DataFrame({
        "entity_key": ["SER_0001", "SER_0002"],
        "entity_name_raw": ["Acme Corp Inc", "Globex Corporation"]
    })
    df_b = pd.DataFrame({
        "entity_key": ["CIK000000001", "CIK000000002"],
        "entity_name_raw": ["Acme Corp", "Globex Corp"]
    })
    cw = build_crosswalk(df_a, df_b, threshold=0.5)
    assert len(cw) == 2
    rep = resolution_report(cw, df_a, df_b)
    assert rep['match_rate'] >= 0.5
    assert rep['resolution_precision'] > 0.0
