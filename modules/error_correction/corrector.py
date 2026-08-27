"""
Database-Aware Error Correction Module
=======================================
Fixes common ASR transcription errors before sending text to the
Text-to-SQL module. Uses fuzzy matching against database terms
(table names, column names, enum values) to correct misheard words.

This is a lightweight, rule-based approach that improves robustness
without requiring additional ML models.
"""

import difflib
import logging
import re
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

# ─── Domain-Specific Replacement Dictionary ──────────────────────────────────
# Maps common spoken variations / ASR misheards to database-correct terms.
DOMAIN_DICTIONARY = {
    # Department names (common ASR errors)
    "computer signs": "Computer Science",
    "computer science": "Computer Science",
    "computer sience": "Computer Science",
    "comp sci": "Computer Science",
    "compsci": "Computer Science",
    "electrical engine": "Electrical Engineering",
    "electric engineering": "Electrical Engineering",
    "english lit": "English Literature",
    "english literature": "English Literature",
    "bio": "Biology",
    "psych": "Psychology",
    "econ": "Economics",
    "math": "Mathematics",
    "maths": "Mathematics",

    # Academic terms
    "grade point average": "GPA",
    "grade point": "GPA",
    "grade-point average": "GPA",
    "enrolment": "enrollment",
    "enrolments": "enrollments",
    "enrolement": "enrollment",
    "enrollment's": "enrollments",
    "proffessor": "professor",
    "proffessors": "professors",
    "professer": "professor",
    "scolarship": "scholarship",
    "scholership": "scholarship",
    "scholarships": "scholarships",
    "full proffessor": "Full Professor",
    "assistant proffessor": "Assistant Professor",
    "associate proffessor": "Associate Professor",

    # Common spoken variations
    "top students": "students with highest GPA",
    "best students": "students with highest GPA",
    "how many": "count",
    "number of": "count",
}


class ErrorCorrector:
    """
    Database-aware error corrector for ASR transcription output.

    Applies three correction strategies in sequence:
    1. Domain dictionary lookup (exact phrase replacement)
    2. Fuzzy matching against database terms (table/column names)
    3. Fuzzy matching against database values (enum values, names)
    """

    def __init__(self, db_terms: Optional[set] = None, similarity_threshold: float = 0.82):
        """
        Initialize the error corrector.

        Args:
            db_terms: Set of known database terms. If None, loads from DB.
            similarity_threshold: Minimum similarity (0-1) for fuzzy matching.
        """
        self.similarity_threshold = similarity_threshold
        self._db_terms = db_terms
        self._db_values = None

    @property
    def db_terms(self) -> set:
        """Lazy-load database terms."""
        if self._db_terms is None:
            try:
                from database.connection import get_all_db_terms
                self._db_terms = get_all_db_terms()
                logger.info(f"Loaded {len(self._db_terms)} database terms for error correction.")
            except Exception as e:
                logger.warning(f"Could not load DB terms: {e}. Using empty set.")
                self._db_terms = set()
        return self._db_terms

    @property
    def db_values(self) -> dict:
        """Lazy-load database values (name→correct_form mapping)."""
        if self._db_values is None:
            self._db_values = {}
            try:
                from database.connection import get_schema_info
                schema = get_schema_info()
                for table_info in schema["tables"].values():
                    for col_name, values in table_info.get("sample_values", {}).items():
                        for val in values:
                            self._db_values[val.lower()] = val
            except Exception as e:
                logger.warning(f"Could not load DB values: {e}")
        return self._db_values

    def correct(self, text: str) -> dict:
        """
        Apply error correction to transcribed text.

        Args:
            text: Raw ASR transcription output.

        Returns:
            dict: {
                "original": str,        # Original text
                "corrected": str,       # Corrected text
                "corrections": list,    # List of corrections made
                "was_corrected": bool   # Whether any corrections were applied
            }
        """
        if not text or not text.strip():
            return {
                "original": text,
                "corrected": text,
                "corrections": [],
                "was_corrected": False,
            }

        original = text
        corrections = []

        # Step 1: Domain dictionary replacement (case-insensitive)
        text, dict_corrections = self._apply_domain_dictionary(text)
        corrections.extend(dict_corrections)

        # Step 2: Fuzzy match individual words against DB terms
        text, fuzzy_corrections = self._apply_fuzzy_matching(text)
        corrections.extend(fuzzy_corrections)

        # Step 3: Try to match multi-word values (e.g., department names)
        text, value_corrections = self._apply_value_matching(text)
        corrections.extend(value_corrections)

        was_corrected = len(corrections) > 0

        if was_corrected:
            logger.info(
                f"Error correction applied {len(corrections)} fix(es): "
                f"'{original}' → '{text}'"
            )

        return {
            "original": original,
            "corrected": text,
            "corrections": corrections,
            "was_corrected": was_corrected,
        }

    def _apply_domain_dictionary(self, text: str) -> tuple[str, list]:
        """Apply domain dictionary replacements."""
        corrections = []
        text_lower = text.lower()

        # Sort by length (longest first) to avoid partial replacements
        sorted_entries = sorted(DOMAIN_DICTIONARY.items(), key=lambda x: len(x[0]), reverse=True)

        for wrong, correct in sorted_entries:
            if wrong.lower() in text_lower:
                # Replace preserving surrounding context
                pattern = re.compile(re.escape(wrong), re.IGNORECASE)
                if pattern.search(text):
                    text = pattern.sub(correct, text)
                    text_lower = text.lower()
                    corrections.append({
                        "type": "domain_dictionary",
                        "from": wrong,
                        "to": correct,
                    })

        return text, corrections

    def _apply_fuzzy_matching(self, text: str) -> tuple[str, list]:
        """Apply fuzzy matching against database terms for individual words."""
        corrections = []
        words = text.split()
        new_words = []

        for word in words:
            clean_word = word.strip(".,!?;:'\"").lower()

            # Skip short words and common English words
            if len(clean_word) <= 3 or clean_word in _COMMON_WORDS:
                new_words.append(word)
                continue

            # Skip if word is already a known DB term
            if clean_word in self.db_terms:
                new_words.append(word)
                continue

            # Find closest match in DB terms
            # Only match terms of similar length (±3 chars) to avoid wild matches
            candidates = [
                t for t in self.db_terms
                if len(t) > 3 and abs(len(t) - len(clean_word)) <= 3
            ]
            matches = difflib.get_close_matches(
                clean_word,
                candidates,
                n=1,
                cutoff=self.similarity_threshold,
            )

            if matches:
                best_match = matches[0]
                # Preserve original punctuation
                prefix = ""
                suffix = ""
                for char in word:
                    if char.isalpha():
                        break
                    prefix += char
                for char in reversed(word):
                    if char.isalpha():
                        break
                    suffix = char + suffix

                new_word = prefix + best_match + suffix
                new_words.append(new_word)
                corrections.append({
                    "type": "fuzzy_match",
                    "from": word,
                    "to": new_word,
                    "similarity": round(
                        difflib.SequenceMatcher(None, clean_word, best_match).ratio(), 2
                    ),
                })
            else:
                new_words.append(word)

        return " ".join(new_words), corrections

    def _apply_value_matching(self, text: str) -> tuple[str, list]:
        """Match multi-word database values (e.g., department names, course titles)."""
        corrections = []
        text_lower = text.lower()

        for val_lower, val_correct in self.db_values.items():
            if len(val_lower) > 5 and val_lower not in text_lower:
                # Check fuzzy match for multi-word values
                # Split into 2-3 word windows
                words = text_lower.split()
                val_words = val_lower.split()
                window_size = len(val_words)

                if window_size < 2:
                    continue

                for i in range(len(words) - window_size + 1):
                    window = " ".join(words[i:i + window_size])
                    similarity = difflib.SequenceMatcher(None, window, val_lower).ratio()

                    if similarity >= self.similarity_threshold:
                        # Replace the window with the correct value
                        original_window = " ".join(text.split()[i:i + window_size])
                        text = text.replace(original_window, val_correct, 1)
                        text_lower = text.lower()
                        corrections.append({
                            "type": "value_match",
                            "from": original_window,
                            "to": val_correct,
                            "similarity": round(similarity, 2),
                        })
                        break

        return text, corrections


# Common English words to skip during fuzzy matching
_COMMON_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "must", "and", "or", "but",
    "not", "no", "yes", "in", "on", "at", "to", "for", "of", "with",
    "by", "from", "up", "out", "if", "about", "into", "over", "after",
    "what", "which", "who", "whom", "this", "that", "these", "those",
    "am", "it", "its", "my", "we", "our", "you", "your", "he", "she",
    "him", "her", "his", "they", "them", "their", "me", "all", "each",
    "every", "both", "few", "more", "most", "other", "some", "such",
    "than", "too", "very", "just", "how", "many", "much", "show",
    "find", "get", "list", "give", "tell", "display", "average", "total",
    "count", "number", "highest", "lowest", "most", "least", "per",
    # Natural language words that should NEVER be fuzzy-matched to DB terms
    "enrolled", "department", "departments", "university", "college",
    "student", "students", "course", "courses", "professor", "professors",
    "scholarship", "scholarships", "enrollment", "enrollments",
    "names", "name", "amounts", "amount", "grades", "grade", "above",
    "below", "greater", "less", "between", "where", "when", "then",
    "there", "here", "only", "also", "like", "want", "need", "know",
    "think", "make", "take", "come", "look", "said", "data", "query",
    "table", "database", "select", "results", "information", "records",
    "calculate", "overall", "specific", "particular", "different",
    "taught", "teaching", "operating", "working", "recipients",
}