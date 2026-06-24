"""Tests for database utility functions."""

import pandas as pd
import pytest
from utils.db import get_connection, load_dataframe, query


SAMPLE_DF = pd.DataFrame({
    "id":    [1, 2, 3],
    "name":  ["Alice", "Bob", "Carol"],
    "score": [85.5, 92.0, 78.3],
})


def test_connection_returns_connection():
    con = get_connection()
    assert con is not None
    con.close()


def test_load_and_query_dataframe():
    con = get_connection()
    load_dataframe(SAMPLE_DF, "test_table", con)
    result = con.execute("SELECT COUNT(*) AS n FROM test_table").df()
    assert result["n"].iloc[0] == 3
    con.close()


def test_query_returns_dataframe():
    con = get_connection()
    load_dataframe(SAMPLE_DF, "test_table2", con)
    result = con.execute("SELECT name, score FROM test_table2 WHERE score > 80").df()
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2
    con.close()


def test_aggregation_query():
    con = get_connection()
    load_dataframe(SAMPLE_DF, "test_agg", con)
    result = con.execute("SELECT ROUND(AVG(score), 2) AS avg_score FROM test_agg").df()
    expected = round((85.5 + 92.0 + 78.3) / 3, 2)
    assert abs(result["avg_score"].iloc[0] - expected) < 0.01
    con.close()


def test_window_function():
    con = get_connection()
    load_dataframe(SAMPLE_DF, "test_window", con)
    result = con.execute("""
        SELECT name, score,
               RANK() OVER (ORDER BY score DESC) AS rnk
        FROM test_window
    """).df()
    assert result[result["name"] == "Bob"]["rnk"].iloc[0] == 1
    con.close()
