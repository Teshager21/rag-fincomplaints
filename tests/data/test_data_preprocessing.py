# tests/test_data_preprocessing.py

import pandas as pd

# import os
# import tempfile
from src.data import data_preprocessing as dp


def test_load_data(tmp_path):
    # Create a mini CSV for testing
    csv_content = """Product,Consumer complaint narrative
Credit card,"This is a complaint about fees."
Personal loan,"I am writing to file a complaint about interest."
Money transfers,"The service is slow."
"""
    test_csv = tmp_path / "test.csv"
    test_csv.write_text(csv_content)

    df = dp.load_data(str(test_csv))
    assert not df.empty
    assert list(df.columns) == ["Product", "Consumer complaint narrative"]
    assert df.shape[0] == 3


def test_filter_data():
    data = {
        "Product": ["Credit card", "Mortgage", "Money transfers"],
        "Consumer complaint narrative": ["Complaint text", None, "Another complaint"],
    }
    df = pd.DataFrame(data)
    filtered = dp.filter_data(df, dp.TARGET_PRODUCTS)
    # Should keep only rows matching target products AND with narratives
    assert all(filtered["Product"].isin(dp.TARGET_PRODUCTS))
    assert filtered.shape[0] == 2
    assert filtered["Consumer complaint narrative"].isna().sum() == 0


def test_clean_text():
    raw_text = "I am writing to file a complaint about charges & fees!"
    cleaned = dp.clean_text(raw_text)
    assert "i am writing" not in cleaned
    assert "&" not in cleaned
    assert "charges" in cleaned
    assert cleaned.islower()


def test_clean_narratives():
    df = pd.DataFrame(
        {
            "Consumer complaint narrative": [
                "I want to file a complaint about this product."
            ]
        }
    )
    cleaned_df = dp.clean_narratives(df)
    assert "Cleaned Narrative" in cleaned_df.columns
    assert "i want to" not in cleaned_df.loc[0, "Cleaned Narrative"]


def test_save_filtered_data(tmp_path):
    df = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
    out_path = tmp_path / "filtered.csv"
    dp.save_filtered_data(df, str(out_path))
    assert out_path.exists()

    loaded = pd.read_csv(out_path)
    assert loaded.shape == df.shape
