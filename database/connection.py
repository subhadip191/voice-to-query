"""
Database Connection Module
==========================
Provides SQLAlchemy engine, session helpers, and schema introspection
utilities used by the Text-to-SQL and Error Correction modules.
"""

from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from config import DATABASE_URL


def get_engine() -> Engine:
    """Create and return a SQLAlchemy engine for the configured database."""
    return create_engine(
        DATABASE_URL,
        echo=False,
        connect_args={"check_same_thread": False},  # SQLite specific
    )


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """Context manager yielding a SQLAlchemy session with auto-commit/rollback."""
    engine = get_engine()
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_schema_info() -> dict:
    """
    Introspect the database and return a structured schema description.

    Returns:
        dict: {
            "tables": {
                "table_name": {
                    "columns": [{"name": str, "type": str, "nullable": bool, "primary_key": bool}],
                    "foreign_keys": [{"column": str, "references": str}],
                    "sample_values": {column_name: [values...]}
                }
            }
        }
    """
    engine = get_engine()
    inspector = inspect(engine)
    schema = {"tables": {}}

    for table_name in inspector.get_table_names():
        columns = []
        for col in inspector.get_columns(table_name):
            columns.append({
                "name": col["name"],
                "type": str(col["type"]),
                "nullable": col.get("nullable", True),
                "primary_key": col.get("autoincrement", False) or col["name"].endswith("_id"),
            })

        # Get foreign keys
        fks = []
        for fk in inspector.get_foreign_keys(table_name):
            for col, ref_col in zip(fk["constrained_columns"], fk["referred_columns"]):
                fks.append({
                    "column": col,
                    "references": f"{fk['referred_table']}.{ref_col}",
                })

        # Get sample values for non-PK text columns (useful for error correction)
        sample_values = {}
        try:
            with engine.connect() as conn:
                for col_info in columns:
                    if "TEXT" in str(col_info["type"]).upper() or "VARCHAR" in str(col_info["type"]).upper():
                        result = conn.execute(
                            text(f"SELECT DISTINCT {col_info['name']} FROM {table_name} LIMIT 20")
                        )
                        vals = [str(row[0]) for row in result if row[0] is not None]
                        if vals:
                            sample_values[col_info["name"]] = vals
        except Exception:
            pass

        schema["tables"][table_name] = {
            "columns": columns,
            "foreign_keys": fks,
            "sample_values": sample_values,
        }

    return schema


def get_schema_ddl() -> str:
    """
    Return the CREATE TABLE DDL statements for all tables.
    Used for injecting schema context into LLM prompts.
    """
    engine = get_engine()
    ddl_statements = []

    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL ORDER BY name")
        )
        for row in result:
            ddl_statements.append(row[0])

    return "\n\n".join(ddl_statements)


def get_all_db_terms() -> set:
    """
    Extract all unique terms from the database: table names, column names,
    and distinct text values. Used for ASR error correction.

    Returns:
        set: All unique database terms (lowercased).
    """
    schema = get_schema_info()
    terms = set()

    for table_name, table_info in schema["tables"].items():
        # Add table name
        terms.add(table_name.lower())

        for col in table_info["columns"]:
            # Add column names
            terms.add(col["name"].lower())

        # Add sample text values
        for col_name, values in table_info.get("sample_values", {}).items():
            for val in values:
                terms.add(val.lower())
                # Also add individual words from multi-word values
                for word in val.split():
                    if len(word) > 2:
                        terms.add(word.lower())

    return terms
