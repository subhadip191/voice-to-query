"""Tests for the database setup and connection modules."""

import sqlite3
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))


class TestDatabaseSetup:
    """Test that the database was set up correctly with all tables and data."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Ensure DB exists before tests."""
        from config import DB_PATH
        self.db_path = str(DB_PATH)
        assert Path(self.db_path).exists(), (
            f"Database not found at {self.db_path}. Run 'python database/setup_db.py' first."
        )

    def _query(self, sql: str) -> list:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        conn.close()
        return rows

    def test_tables_exist(self):
        """All 6 tables should exist."""
        tables = [r[0] for r in self._query(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )]
        expected = {"courses", "departments", "enrollments", "professors", "scholarships", "students"}
        assert expected.issubset(set(tables)), f"Missing tables: {expected - set(tables)}"

    def test_departments_count(self):
        assert self._query("SELECT COUNT(*) FROM departments")[0][0] == 8

    def test_professors_count(self):
        assert self._query("SELECT COUNT(*) FROM professors")[0][0] == 20

    def test_students_count(self):
        assert self._query("SELECT COUNT(*) FROM students")[0][0] == 50

    def test_courses_count(self):
        assert self._query("SELECT COUNT(*) FROM courses")[0][0] == 30

    def test_enrollments_count(self):
        assert self._query("SELECT COUNT(*) FROM enrollments")[0][0] == 150

    def test_scholarships_count(self):
        assert self._query("SELECT COUNT(*) FROM scholarships")[0][0] == 20

    def test_join_query(self):
        """Test a JOIN query works correctly."""
        rows = self._query("""
            SELECT s.first_name, d.name
            FROM students s
            JOIN departments d ON s.department_id = d.department_id
            LIMIT 5
        """)
        assert len(rows) == 5
        assert all(len(row) == 2 for row in rows)

    def test_aggregation_query(self):
        """Test an aggregation query."""
        rows = self._query("""
            SELECT d.name, ROUND(AVG(s.gpa), 2)
            FROM students s
            JOIN departments d ON s.department_id = d.department_id
            GROUP BY d.name
        """)
        assert len(rows) == 8  # 8 departments

    def test_subquery(self):
        """Test a subquery."""
        rows = self._query("""
            SELECT first_name, last_name, gpa
            FROM students
            WHERE gpa > (SELECT AVG(gpa) FROM students)
        """)
        assert len(rows) > 0


class TestConnection:
    """Test the connection module."""

    def test_get_schema_info(self):
        from database.connection import get_schema_info
        schema = get_schema_info()
        assert "tables" in schema
        assert "students" in schema["tables"]
        assert "columns" in schema["tables"]["students"]

    def test_get_schema_ddl(self):
        from database.connection import get_schema_ddl
        ddl = get_schema_ddl()
        assert "CREATE TABLE" in ddl
        assert "students" in ddl

    def test_get_all_db_terms(self):
        from database.connection import get_all_db_terms
        terms = get_all_db_terms()
        assert len(terms) > 20
        assert "students" in terms
        assert "computer science" in terms or "computer" in terms
