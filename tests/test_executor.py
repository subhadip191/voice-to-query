"""Tests for the SQL query executor module."""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))


class TestQueryRunner:
    """Test the query execution module."""

    @pytest.fixture
    def runner(self):
        from modules.executor.query_runner import QueryRunner
        return QueryRunner()

    def test_simple_select(self, runner):
        result = runner.execute("SELECT * FROM students LIMIT 5")
        assert result["success"]
        assert result["row_count"] == 5
        assert "first_name" in result["column_names"]

    def test_join_query(self, runner):
        result = runner.execute("""
            SELECT s.first_name, d.name AS department
            FROM students s
            JOIN departments d ON s.department_id = d.department_id
            LIMIT 3
        """)
        assert result["success"]
        assert result["row_count"] == 3
        assert "department" in result["column_names"]

    def test_aggregation(self, runner):
        result = runner.execute("""
            SELECT d.name, COUNT(*) AS count
            FROM students s
            JOIN departments d ON s.department_id = d.department_id
            GROUP BY d.name
        """)
        assert result["success"]
        assert result["row_count"] == 8

    def test_blocks_insert(self, runner):
        result = runner.execute("INSERT INTO students VALUES (999, 'Bad', 'Actor', 'x', '2000-01-01', 'Male', 1, 2024, 4.0)")
        assert not result["success"]
        assert "SELECT" in result["error"]

    def test_blocks_drop(self, runner):
        result = runner.execute("DROP TABLE students")
        assert not result["success"]

    def test_empty_query(self, runner):
        result = runner.execute("")
        assert not result["success"]
        assert "No SQL" in result["error"]

    def test_invalid_table(self, runner):
        result = runner.execute("SELECT * FROM nonexistent_table")
        assert not result["success"]
        assert "not found" in result["error"].lower() or "no such table" in result["error"].lower()

    def test_returns_dataframe(self, runner):
        import pandas as pd
        result = runner.execute("SELECT * FROM departments")
        assert isinstance(result["data"], pd.DataFrame)

    def test_execution_time_tracked(self, runner):
        result = runner.execute("SELECT * FROM students")
        assert result["execution_time_ms"] >= 0

    def test_table_preview(self, runner):
        df = runner.get_table_preview("departments", limit=3)
        assert len(df) == 3
