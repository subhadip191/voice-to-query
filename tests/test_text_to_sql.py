"""Tests for the Text-to-SQL generator module (requires OpenAI API key)."""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))


class TestSchemaPrompt:
    """Test the schema prompt builder (no API key needed)."""

    def test_build_system_prompt(self):
        from modules.text_to_sql.schema_prompt import build_system_prompt
        prompt = build_system_prompt()
        assert "CREATE TABLE" in prompt
        assert "students" in prompt
        assert "departments" in prompt
        assert "SELECT" in prompt  # Few-shot examples

    def test_build_user_prompt(self):
        from modules.text_to_sql.schema_prompt import build_user_prompt
        prompt = build_user_prompt("Show all students")
        assert "Show all students" in prompt


class TestTextToSQLGenerator:
    """Test the Text-to-SQL generator (some tests require API key)."""

    @pytest.fixture
    def generator(self):
        from modules.text_to_sql.generator import TextToSQLGenerator
        return TextToSQLGenerator()

    def test_clean_sql_removes_fences(self, generator):
        raw = "```sql\nSELECT * FROM students;\n```"
        cleaned = generator._clean_sql(raw)
        assert cleaned == "SELECT * FROM students"
        assert "```" not in cleaned

    def test_validate_sql_safety_select(self, generator):
        is_safe, msg = generator._validate_sql_safety("SELECT * FROM students")
        assert is_safe

    def test_validate_sql_safety_with_cte(self, generator):
        is_safe, msg = generator._validate_sql_safety(
            "WITH cte AS (SELECT * FROM students) SELECT * FROM cte"
        )
        assert is_safe

    def test_validate_sql_safety_blocks_drop(self, generator):
        is_safe, msg = generator._validate_sql_safety("DROP TABLE students")
        assert not is_safe

    def test_validate_sql_safety_blocks_insert(self, generator):
        is_safe, msg = generator._validate_sql_safety(
            "INSERT INTO students VALUES (1, 'Test', 'User')"
        )
        assert not is_safe

    def test_validate_sql_safety_blocks_delete(self, generator):
        is_safe, msg = generator._validate_sql_safety("DELETE FROM students WHERE student_id = 1")
        assert not is_safe

    def test_validate_sql_safety_blocks_update(self, generator):
        is_safe, msg = generator._validate_sql_safety("UPDATE students SET gpa = 4.0")
        assert not is_safe

    def test_empty_query(self, generator):
        result = generator.generate_sql("")
        assert result["error"]
        assert not result["is_safe"]

    @pytest.mark.skipif(
        not Path(PROJECT_ROOT / ".env").exists(),
        reason="No .env file (API key needed for live test)"
    )
    def test_live_generation(self, generator):
        """Integration test — requires a valid OPENAI_API_KEY in .env."""
        result = generator.generate_sql("Show all students in Computer Science")
        if "API key" not in (result.get("error") or ""):
            assert result["sql"]
            assert "SELECT" in result["sql"].upper()
            assert result["is_safe"]
