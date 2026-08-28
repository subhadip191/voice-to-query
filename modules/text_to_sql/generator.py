"""
Text-to-SQL Generator
======================
Translates natural language queries into executable SQL using an LLM
(Groq Llama 3.3 70B — free tier, OpenAI-compatible API).
"""

import logging
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL

logger = logging.getLogger(__name__)


class TextToSQLGenerator:
    """
    Converts natural language questions into SQL queries using an LLM.

    The generator uses schema-aware prompting to ensure the LLM understands
    the database structure and generates accurate, executable SQL.
    """

    # SQL keywords that indicate dangerous write operations
    FORBIDDEN_KEYWORDS = [
        "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE",
        "TRUNCATE", "REPLACE", "GRANT", "REVOKE", "EXEC",
    ]

    def __init__(self, api_key: str = None, base_url: str = None, model: str = None):
        """
        Initialize the Text-to-SQL generator.

        Args:
            api_key: LLM API key. Falls back to config/env (GROQ_API_KEY).
            base_url: LLM API base URL. Falls back to config (Groq endpoint).
            model: LLM model name. Falls back to config (llama-3.3-70b-versatile).
        """
        self.api_key = api_key or LLM_API_KEY
        self.base_url = base_url or LLM_BASE_URL
        self.model = model or LLM_MODEL
        self._client = None
        self._system_prompt = None

    @property
    def client(self):
        """Lazy-initialize the OpenAI-compatible client (pointing to Groq)."""
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
        return self._client

    @property
    def system_prompt(self):
        """Lazy-load the schema-aware system prompt."""
        if self._system_prompt is None:
            from modules.text_to_sql.schema_prompt import build_system_prompt
            self._system_prompt = build_system_prompt()
        return self._system_prompt

    def generate_sql(self, natural_language: str) -> dict:
        """
        Convert a natural language question into an SQL query.

        Args:
            natural_language: The user's question in plain English.

        Returns:
            dict: {
                "sql": str,            # Generated SQL query
                "explanation": str,    # Brief explanation of the query
                "is_safe": bool,       # Whether query passed safety check
                "error": str | None    # Error message if any
            }
        """
        if not natural_language or not natural_language.strip():
            return {
                "sql": "",
                "explanation": "",
                "is_safe": False,
                "error": "Empty query provided.",
            }

        if not self.api_key:
            return {
                "sql": "",
                "explanation": "",
                "is_safe": False,
                "error": "LLM API key not configured. Set GROQ_API_KEY in .env file (free at console.groq.com).",
            }

        try:
            # Call LLM for SQL generation
            from modules.text_to_sql.schema_prompt import build_user_prompt
            user_msg = build_user_prompt(natural_language)

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,  # Deterministic output for SQL
                max_tokens=500,
            )

            raw_sql = response.choices[0].message.content.strip()

            # Clean the SQL output
            sql = self._clean_sql(raw_sql)

            # Safety validation
            is_safe, safety_msg = self._validate_sql_safety(sql)

            if not is_safe:
                logger.warning(f"Unsafe SQL detected: {safety_msg}")
                return {
                    "sql": sql,
                    "explanation": "",
                    "is_safe": False,
                    "error": f"Safety check failed: {safety_msg}",
                }

            # Generate a brief explanation
            explanation = self._generate_explanation(natural_language, sql)

            logger.info(f"Generated SQL for '{natural_language[:60]}': {sql[:100]}")

            return {
                "sql": sql,
                "explanation": explanation,
                "is_safe": True,
                "error": None,
            }

        except Exception as e:
            logger.error(f"SQL generation failed: {e}")
            return {
                "sql": "",
                "explanation": "",
                "is_safe": False,
                "error": f"SQL generation error: {str(e)}",
            }

    def _clean_sql(self, raw_sql: str) -> str:
        """
        Clean LLM output by removing markdown code fences and extra whitespace.

        Args:
            raw_sql: Raw LLM output.

        Returns:
            str: Cleaned SQL query.
        """
        # Remove markdown code fences: ```sql ... ``` or ``` ... ```
        sql = re.sub(r"```(?:sql)?\s*", "", raw_sql)
        sql = sql.replace("```", "")

        # Remove leading/trailing whitespace and newlines
        sql = sql.strip()

        # Remove trailing semicolons (SQLAlchemy handles this)
        sql = sql.rstrip(";")

        return sql

    def _validate_sql_safety(self, sql: str) -> tuple[bool, str]:
        """
        Validate that the SQL query is read-only (SELECT only).

        Args:
            sql: The SQL query to validate.

        Returns:
            tuple: (is_safe: bool, message: str)
        """
        if not sql:
            return False, "Empty SQL query."

        sql_upper = sql.upper().strip()

        # Must start with SELECT or WITH (for CTEs)
        if not sql_upper.startswith(("SELECT", "WITH")):
            return False, "Query must be a SELECT statement."

        # Check for forbidden keywords
        for keyword in self.FORBIDDEN_KEYWORDS:
            # Use word boundary matching to avoid false positives
            pattern = r"\b" + keyword + r"\b"
            if re.search(pattern, sql_upper):
                return False, f"Forbidden keyword detected: {keyword}"

        return True, "Query is safe."

    def _generate_explanation(self, question: str, sql: str) -> str:
        """
        Generate a brief, human-readable explanation of the SQL query.

        Args:
            question: Original natural language question.
            sql: Generated SQL query.

        Returns:
            str: Brief explanation.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful assistant. Given a SQL query and the original question, "
                            "provide a brief one-sentence explanation of what the query does. "
                            "Be concise — max 20 words."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Question: {question}\nSQL: {sql}",
                    },
                ],
                temperature=0.0,
                max_tokens=60,
            )
            return response.choices[0].message.content.strip()
        except Exception:
            return "Executes a database query based on your question."
