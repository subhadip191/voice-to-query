"""
Query Execution Module
=======================
Executes validated SQL queries against the database and returns
results as pandas DataFrames with error handling and timeout protection.
"""

import logging
import sqlite3
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from config import DB_PATH, MAX_RESULT_ROWS, QUERY_TIMEOUT_SECONDS

logger = logging.getLogger(__name__)


class QueryRunner:
    """
    Executes SQL queries safely against the SQLite database.

    Features:
    - Read-only enforcement
    - Timeout protection
    - Result capping
    - User-friendly error messages
    """

    def __init__(self, db_path: str = None, timeout: int = None, max_rows: int = None):
        """
        Initialize the query runner.

        Args:
            db_path: Path to SQLite database file.
            timeout: Query execution timeout in seconds.
            max_rows: Maximum number of result rows to return.
        """
        self.db_path = str(db_path or DB_PATH)
        self.timeout = timeout or QUERY_TIMEOUT_SECONDS
        self.max_rows = max_rows or MAX_RESULT_ROWS

    def execute(self, sql: str) -> dict:
        """
        Execute a SQL query and return results as a DataFrame.

        Args:
            sql: The SQL query to execute (must be SELECT/WITH only).

        Returns:
            dict: {
                "success": bool,
                "data": pd.DataFrame,      # Query results
                "row_count": int,           # Number of rows returned
                "column_names": list,       # Column names
                "truncated": bool,          # Whether results were capped
                "error": str | None,        # Error message if any
                "execution_time_ms": float  # Execution time in milliseconds
            }
        """
        if not sql or not sql.strip():
            return self._error_result("No SQL query provided.")

        # Final safety check — only allow SELECT and WITH
        sql_upper = sql.strip().upper()
        if not sql_upper.startswith(("SELECT", "WITH")):
            return self._error_result(
                "Only SELECT queries are allowed. "
                "The query was blocked for safety."
            )

        start_time = time.time()

        try:
            conn = sqlite3.connect(self.db_path, timeout=self.timeout)
            conn.row_factory = sqlite3.Row

            # Set a busy timeout
            conn.execute(f"PRAGMA busy_timeout = {self.timeout * 1000};")

            cursor = conn.cursor()
            cursor.execute(sql)

            # Fetch results
            rows = cursor.fetchmany(self.max_rows + 1)
            truncated = len(rows) > self.max_rows
            if truncated:
                rows = rows[:self.max_rows]

            # Get column names and deduplicate them for PyArrow/Streamlit compatibility
            raw_column_names = [description[0] for description in cursor.description] if cursor.description else []
            column_names = []
            seen_counts = {}
            for col in raw_column_names:
                if col in seen_counts:
                    seen_counts[col] += 1
                    column_names.append(f"{col}_{seen_counts[col]}")
                else:
                    seen_counts[col] = 0
                    column_names.append(col)

            # Convert to DataFrame
            data = pd.DataFrame(
                [dict(zip(column_names, row)) for row in rows],
                columns=column_names
            )

            elapsed_ms = round((time.time() - start_time) * 1000, 2)

            logger.info(
                f"Query executed: {len(data)} rows in {elapsed_ms}ms"
            )

            conn.close()

            return {
                "success": True,
                "data": data,
                "row_count": len(data),
                "column_names": column_names,
                "truncated": truncated,
                "error": None,
                "execution_time_ms": elapsed_ms,
            }

        except sqlite3.OperationalError as e:
            elapsed_ms = round((time.time() - start_time) * 1000, 2)
            error_msg = str(e)

            # Provide user-friendly error messages
            if "no such table" in error_msg:
                friendly = f"Table not found: {error_msg}. Please check the table name."
            elif "no such column" in error_msg:
                friendly = f"Column not found: {error_msg}. Please check the column name."
            elif "ambiguous column" in error_msg:
                friendly = f"Ambiguous column: {error_msg}. Use table aliases to qualify the column."
            elif "near" in error_msg:
                friendly = f"SQL syntax error: {error_msg}."
            else:
                friendly = f"Database error: {error_msg}"

            logger.error(f"Query execution failed ({elapsed_ms}ms): {error_msg}")
            return self._error_result(friendly, elapsed_ms)

        except Exception as e:
            elapsed_ms = round((time.time() - start_time) * 1000, 2)
            logger.error(f"Unexpected error ({elapsed_ms}ms): {e}")
            return self._error_result(f"Unexpected error: {str(e)}", elapsed_ms)

    def _error_result(self, error_msg: str, elapsed_ms: float = 0.0) -> dict:
        """Create a standardized error result."""
        return {
            "success": False,
            "data": pd.DataFrame(),
            "row_count": 0,
            "column_names": [],
            "truncated": False,
            "error": error_msg,
            "execution_time_ms": elapsed_ms,
        }

    def get_table_preview(self, table_name: str, limit: int = 5) -> pd.DataFrame:
        """
        Get a preview of a database table.

        Args:
            table_name: Name of the table to preview.
            limit: Number of rows to return.

        Returns:
            pd.DataFrame: Preview of the table.
        """
        # Sanitize table name to prevent injection
        safe_name = "".join(c for c in table_name if c.isalnum() or c == "_")
        result = self.execute(f"SELECT * FROM {safe_name} LIMIT {limit}")
        return result["data"]
