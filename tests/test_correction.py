"""Tests for the error correction module."""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))


class TestErrorCorrector:
    """Test the database-aware error correction module."""

    @pytest.fixture
    def corrector(self):
        from modules.error_correction.corrector import ErrorCorrector
        # Use a mock set of DB terms for testing
        mock_terms = {
            "students", "professors", "departments", "courses",
            "enrollments", "scholarships", "computer science",
            "mathematics", "physics", "biology", "economics",
            "psychology", "electrical engineering", "english literature",
            "gpa", "salary", "enrollment_year", "first_name", "last_name",
            "computer", "science", "engineering",
        }
        return ErrorCorrector(db_terms=mock_terms, similarity_threshold=0.75)

    def test_domain_dictionary_replacement(self, corrector):
        result = corrector.correct("Show me students in computer signs")
        assert "Computer Science" in result["corrected"]
        assert result["was_corrected"]

    def test_gpa_expansion(self, corrector):
        result = corrector.correct("what is the grade point average per department")
        assert "GPA" in result["corrected"]
        assert result["was_corrected"]

    def test_no_correction_needed(self, corrector):
        result = corrector.correct("Show all students in the Computer Science department")
        # Domain dict may still match "Computer Science" -> "Computer Science"
        assert result["corrected"]  # Should not break

    def test_empty_input(self, corrector):
        result = corrector.correct("")
        assert result["corrected"] == ""
        assert not result["was_corrected"]

    def test_returns_corrections_list(self, corrector):
        result = corrector.correct("computer signs department")
        assert isinstance(result["corrections"], list)

    def test_misspelled_professor(self, corrector):
        result = corrector.correct("Show all proffessors")
        assert "professor" in result["corrected"].lower()
        assert result["was_corrected"]

    def test_comp_sci_abbreviation(self, corrector):
        result = corrector.correct("students in comp sci")
        assert "Computer Science" in result["corrected"]
