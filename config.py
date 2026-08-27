"""
Voice2Query — Central Configuration
====================================
Loads settings from .env file and provides project-wide constants.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ─── Load Environment Variables ─────────────────────────────────────────────
load_dotenv(override=True)

# ─── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.resolve()
DB_PATH = PROJECT_ROOT / os.getenv("DB_PATH", "voice2query.db")
AUDIO_SAMPLES_DIR = PROJECT_ROOT / "audio_samples"
SCHEMA_SQL_PATH = PROJECT_ROOT / "database" / "schema.sql"
SEED_SQL_PATH = PROJECT_ROOT / "database" / "seed_data.sql"

# ─── Database ───────────────────────────────────────────────────────────────
DATABASE_URL = f"sqlite:///{DB_PATH}"

# ─── LLM Provider (Groq — free tier, OpenAI-compatible API) ─────────────────
LLM_API_KEY = os.getenv("GROQ_API_KEY", os.getenv("OPENAI_API_KEY", ""))
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

# ─── Whisper (ASR) ──────────────────────────────────────────────────────────
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")

# ─── Audio Recording Settings ──────────────────────────────────────────────
AUDIO_SAMPLE_RATE = 16000   # Hz — Whisper expects 16kHz
AUDIO_CHANNELS = 1          # Mono
MAX_RECORDING_SECONDS = 30  # Maximum recording duration

# ─── Query Execution ───────────────────────────────────────────────────────
QUERY_TIMEOUT_SECONDS = 5   # Max execution time for SQL queries
MAX_RESULT_ROWS = 500       # Cap results to prevent memory issues
