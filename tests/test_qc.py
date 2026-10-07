import pandas as pd
from vtn.qc import check_schema, check_duplicate_keys, run_all

def test_qc_gate(tmp_path):
    df_good = pd.DataFrame({
        "entity_id": ["ENT_001", "ENT_002"],
        "date": ["2022-01-01", "2022-01-02"],
        "value_a": [100.0, 200.0]
    })
    
    res_schema = check_schema(df_good)
    assert res_schema['passed'] is True
    
    res_dups = check_duplicate_keys(df_good)
    assert res_dups['passed'] is True
    
    passed, checks = run_all(df_good, out_path=tmp_path / "good.json")
    assert passed is True
    
    # Bad frame with duplicate keys
    df_bad = pd.DataFrame({
        "entity_id": ["ENT_001", "ENT_001"],
        "date": ["2022-01-01", "2022-01-01"],
        "value_a": [100.0, 150.0]
    })
    passed_bad, _ = run_all(df_bad, out_path=tmp_path / "bad.json")
    assert passed_bad is False
