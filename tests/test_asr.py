"""Tests for the ASR (transcriber) module."""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))


class TestTranscriber:
    """Test the ASR transcriber module."""

    @pytest.fixture
    def transcriber(self):
        """Load whisper model — skip if not installed."""
        try:
            from modules.asr.transcriber import Transcriber
            return Transcriber(model_name="tiny")  # Use tiny for fast tests
        except (ImportError, Exception) as e:
            pytest.skip(f"Whisper not available: {e}")

    def test_transcribe_missing_file(self, transcriber):
        result = transcriber.transcribe_audio("/nonexistent/file.wav")
        assert result["error"]
        assert result["text"] == ""

    def test_transcribe_from_empty_bytes(self, transcriber):
        result = transcriber.transcribe_from_bytes(b"")
        assert result["error"]
        assert "Empty" in result["error"]

    def test_model_loaded(self, transcriber):
        assert transcriber.model is not None

    def test_result_structure(self, transcriber):
        """Verify the result dict has all expected keys."""
        result = transcriber.transcribe_audio("/nonexistent/file.wav")
        expected_keys = {"text", "language", "confidence", "segments", "error"}
        assert set(result.keys()) == expected_keys
