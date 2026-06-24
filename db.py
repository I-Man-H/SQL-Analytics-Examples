"""
Database utilities.

Uses DuckDB as the analytical engine — it runs fully in-process,
requires no server setup, and supports the full SQL analytics
feature set (window functions, CTEs, PIVOT, UNNEST, etc.).

All notebooks and scripts import from here so the connection
logic lives in one place.
"""

import os
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd


DB_PATH = os.environ.get("DUCKDB_PATH", ":memory:")


def get_connection(db_path: str = DB_PATH) -> duckdb.DuckDBPyConnection:
    """
    Return a DuckDB connection.

    Uses an in-memory database by default (fast, no files left behind).
    Set DUCKDB_PATH env var or pass a path to persist the database to disk.
    """
    return duckdb.connect(db_path)


def query(sql: str, db_path: str = DB_PATH) -> pd.DataFrame:
    """
    Execute a SQL query and return results as a DataFrame.

    Args:
        sql:     SQL string to execute.
        db_path: DuckDB database path (default: in-memory).

    Returns:
        pandas DataFrame with query results.
    """
    with get_connection(db_path) as con:
        return con.execute(sql).df()


def load_csv(
    filepath: str,
    table_name: str,
    con: Optional[duckdb.DuckDBPyConnection] = None,
) -> None:
    """
    Load a CSV file directly into a DuckDB table.

    DuckDB can read CSVs with automatic type inference — no manual
    schema definition required for exploratory work.

    Args:
        filepath:   Path to the CSV file.
        table_name: Name of the table to create.
        con:        Existing connection (creates a new one if None).
    """
    own_con = con is None
    if own_con:
        con = get_connection()

    con.execute(
        f"CREATE OR REPLACE TABLE {table_name} AS "
        f"SELECT * FROM read_csv_auto('{filepath}')"
    )

    if own_con:
        con.close()


def load_dataframe(
    df: pd.DataFrame,
    table_name: str,
    con: duckdb.DuckDBPyConnection,
) -> None:
    """
    Register a pandas DataFrame as a DuckDB table.

    Useful when data has already been loaded or synthesised in Python.
    The registration is zero-copy — DuckDB reads directly from the
    DataFrame's memory without duplication.

    Args:
        df:         Source DataFrame.
        table_name: Table name to register under.
        con:        Active DuckDB connection.
    """
    con.register(table_name, df)


def run_sql_file(filepath: str, con: duckdb.DuckDBPyConnection) -> None:
    """
    Execute all statements in a .sql file against an open connection.

    Args:
        filepath: Path to the .sql file.
        con:      Active DuckDB connection.
    """
    sql = Path(filepath).read_text()
    # Split on semicolons, skip empty statements
    statements = [s.strip() for s in sql.split(";") if s.strip()]
    for stmt in statements:
        con.execute(stmt)
