"""
ASR (Automatic Speech Recognition) Module
==========================================
Transcribes audio files to text using OpenAI's Whisper model (locally deployed).
Handles acoustic variability, format conversion, and confidence assessment.

Supports: .wav, .mp3, .m4a, .webm, .flac, .ogg
"""

import logging
import math
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class Transcriber:
    """
    Speech-to-text transcriber using OpenAI Whisper.

    Attributes:
        model_name: Whisper model size (tiny, base, small, medium, large).
        model: Loaded Whisper model instance.
    """

    def __init__(self, model_name: str = "base"):
        """
        Initialize the transcriber with the specified Whisper model.

        Args:
            model_name: Model size. Options: tiny, base, small, medium, large.
                        Larger models are more accurate but slower.
        """
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the Whisper model. Downloads on first run."""
        try:
            import whisper
            logger.info(f"Loading Whisper model: {self.model_name}")
            self.model = whisper.load_model(self.model_name)
            logger.info(f"Whisper model '{self.model_name}' loaded successfully.")
        except ImportError:
            logger.error(
                "openai-whisper is not installed. "
                "Install it with: pip install openai-whisper"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise

    def transcribe_audio(
        self,
        file_path: str,
        language: Optional[str] = "en",
    ) -> dict:
        """
        Transcribe an audio file to text.

        Args:
            file_path: Path to the audio file (.wav, .mp3, .m4a, etc.).
            language: Language code (e.g., 'en'). None for auto-detection.

        Returns:
            dict: {
                "text": str,           # Transcribed text
                "language": str,       # Detected language
                "confidence": float,   # Estimated confidence (0-1)
                "segments": list,      # Detailed segment info
                "error": str | None    # Error message if any
            }
        """
        if self.model is None:
            return {
                "text": "",
                "language": "",
                "confidence": 0.0,
                "segments": [],
                "error": "Whisper model not loaded.",
            }

        file_path = str(file_path)

        # Validate file exists
        if not Path(file_path).exists():
            return {
                "text": "",
                "language": "",
                "confidence": 0.0,
                "segments": [],
                "error": f"Audio file not found: {file_path}",
            }

        try:
            # Convert to wav if needed using pydub
            file_path = self._ensure_compatible_format(file_path)

            # Transcribe
            result = self.model.transcribe(
                file_path,
                language=language,
                fp16=False,  # Use FP32 for CPU compatibility
            )

            text = result.get("text", "").strip()

            # Check for empty transcription (silence detection)
            if not text:
                return {
                    "text": "",
                    "language": result.get("language", ""),
                    "confidence": 0.0,
                    "segments": [],
                    "error": "No speech detected in the audio. The recording may be silent or too noisy.",
                }

            # Calculate average confidence from segments
            segments = result.get("segments", [])
            if segments:
                avg_confidence = sum(
                    seg.get("avg_logprob", -1) for seg in segments
                ) / len(segments)
                # Convert log probability to a 0-1 confidence score
                confidence = min(1.0, max(0.0, math.exp(avg_confidence)))
            else:
                confidence = 0.5  # Default if no segment info

            logger.info(
                f"Transcription complete: '{text[:80]}...' "
                f"(confidence: {confidence:.2f}, language: {result.get('language', 'unknown')})"
            )

            return {
                "text": text,
                "language": result.get("language", "en"),
                "confidence": round(confidence, 3),
                "segments": [
                    {
                        "start": seg.get("start", 0),
                        "end": seg.get("end", 0),
                        "text": seg.get("text", ""),
                    }
                    for seg in segments
                ],
                "error": None,
            }

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return {
                "text": "",
                "language": "",
                "confidence": 0.0,
                "segments": [],
                "error": f"Transcription error: {str(e)}",
            }

    def transcribe_from_bytes(
        self,
        audio_bytes: bytes,
        file_extension: str = ".wav",
        language: Optional[str] = "en",
    ) -> dict:
        """
        Transcribe audio from raw bytes (e.g., from Streamlit audio input).

        Args:
            audio_bytes: Raw audio data.
            file_extension: File extension for temp file (e.g., '.wav', '.webm').
            language: Language code. None for auto-detection.

        Returns:
            dict: Same structure as transcribe_audio().
        """
        if not audio_bytes:
            return {
                "text": "",
                "language": "",
                "confidence": 0.0,
                "segments": [],
                "error": "Empty audio data received.",
            }

        # Write bytes to a temporary file
        with tempfile.NamedTemporaryFile(
            suffix=file_extension, delete=False
        ) as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name

        try:
            result = self.transcribe_audio(tmp_path, language=language)
        finally:
            # Clean up temp file
            Path(tmp_path).unlink(missing_ok=True)

        return result

    def _ensure_compatible_format(self, file_path: str) -> str:
        """
        Convert audio to WAV format if it's not already compatible.
        Whisper handles most formats, but this ensures edge-case compatibility.

        Args:
            file_path: Path to the original audio file.

        Returns:
            str: Path to the (possibly converted) audio file.
        """
        ext = Path(file_path).suffix.lower()
        compatible_formats = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

        if ext in compatible_formats:
            return file_path

        try:
            from pydub import AudioSegment

            logger.info(f"Converting {ext} to .wav for compatibility...")
            audio = AudioSegment.from_file(file_path)
            wav_path = tempfile.mktemp(suffix=".wav")
            audio.export(wav_path, format="wav")
            return wav_path
        except ImportError:
            logger.warning("pydub not installed; attempting direct transcription.")
            return file_path
        except Exception as e:
            logger.warning(f"Audio conversion failed ({e}); attempting direct transcription.")
            return file_path
