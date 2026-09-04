# Voice2Query: AI-Powered Speech-to-SQL for Interactive Database Exploration

**Final Project — Data Management, 2026 @ UNINA**

---

## Group Members

| Name | Student ID |
|------|------------|
| Subhadip Maity | D03000291 |
| Vedant Gajanan Pawar | D03000257 |
| Vishal Kumar | D03000263 |
| Deepak Kushwaha | D03000258 |

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [System Architecture](#2-system-architecture)
3. [Task 1 — Speech Recognition (ASR)](#3-task-1--speech-recognition-asr)
4. [Task 2 — Text-to-SQL Translation](#4-task-2--text-to-sql-translation)
5. [Database Design](#5-database-design)
6. [Pipeline Integration](#6-pipeline-integration)
7. [Error Correction Module](#7-error-correction-module)
8. [Interactive Dashboard](#8-interactive-dashboard)
9. [Evaluation and Results](#9-evaluation-and-results)
10. [Related Work Discussion](#10-related-work-discussion)
11. [Limitations and Future Work](#11-limitations-and-future-work)
12. [Conclusion](#12-conclusion)

---

## 1. Introduction

The goal of this project is to design and implement a **Speech-to-SQL pipeline** that allows users to interact with a relational database using natural spoken language — without requiring any knowledge of SQL. This aligns with the broader field of Natural Language Interfaces to Databases (NLIDBs), which has evolved from early rule-based systems to modern neural and LLM-based architectures.

The resulting system, **Voice2Query**, implements a cascaded pipeline architecture where:

1. The user speaks a question (or types/uploads audio)
2. An ASR module transcribes the speech to text
3. A database-aware error correction module cleans ASR errors
4. An LLM translates the corrected text into a SQL query
5. The query is safely executed on a SQLite university database
6. Results are visualised through an interactive Streamlit dashboard

The system integrates **Speech Recognition (ASR)**, **Natural Language Processing (NLP)**, **Text-to-SQL via LLMs**, and **data visualisation** — the four pillars specified in the project brief.

---

## 2. System Architecture

The pipeline follows a cascaded design, which is the dominant paradigm in the literature for practical deployment (Song et al., 2022 [4]). The architecture is:

```
┌──────────┐     ┌─────────────┐     ┌──────────────┐     ┌───────────────┐     ┌─────────────┐
│  Audio   │ ──▸ │ ASR Module  │ ──▸ │   Error      │ ──▸ │  Text-to-SQL  │ ──▸ │  Execute &  │
│  Input   │     │  (Whisper)  │     │  Correction  │     │ (Groq Llama)  │     │  Visualise  │
└──────────┘     └─────────────┘     └──────────────┘     └───────────────┘     └─────────────┘
                                             ▲                     ▲
                                             │                     │
                                   ┌─────────┴─────────────────────┴──────┐
                                   │      SQLite University Database       │
                                   │    (schema DDL + sample values)       │
                                   └───────────────────────────────────────┘
```

**Key design decisions:**

- **Cascaded over end-to-end**: We chose a cascaded pipeline rather than an end-to-end speech-to-SQL model (as in SpeechSQLNet [2] or Wav2SQL [3]). End-to-end models require large paired speech-SQL training corpora and specialised hardware, making them impractical for this project. The cascaded approach allows us to use state-of-the-art, freely available components (Whisper + Groq Llama) and provides interpretability at each stage — essential for debugging and demonstration.

- **SQLite as DBMS**: Chosen for zero-configuration deployment, single-file portability, and full SQL feature support (JOINs, aggregations, subqueries, triggers). It is appropriate for a university-scale dataset and removes infrastructure complexity.

- **Groq API for LLM inference**: Free-tier access to Llama 3.3 70B with low latency (~500–1500ms), OpenAI-compatible API, and no local GPU requirement.

### Project Structure

```
voice2query/
├── config.py                          # Centralised configuration & env vars
├── evaluate.py                        # End-to-end evaluation script
├── requirements.txt
├── database/
│   ├── schema.sql                     # DDL: 6 tables + triggers
│   ├── seed_data.sql                  # 250+ rows of realistic mock data
│   ├── setup_db.py                    # DB initialisation script
│   └── connection.py                  # SQLAlchemy + schema introspection
├── modules/
│   ├── asr/transcriber.py             # Whisper speech-to-text
│   ├── text_to_sql/
│   │   ├── generator.py               # LLM-based NL → SQL
│   │   └── schema_prompt.py           # Schema-aware prompt builder
│   ├── error_correction/corrector.py  # DB-aware fuzzy error correction
│   └── executor/query_runner.py       # Safe SQL execution engine
├── dashboard/app.py                   # Streamlit interactive UI
└── tests/                             # Pytest suite (5 modules)
```

---

## 3. Task 1 — Speech Recognition (ASR)

### 3.1 Tool Comparison: Whisper vs Speechmatics

We analysed the two recommended ASR tools along five dimensions:

| Dimension | **OpenAI Whisper** | **Speechmatics** |
|---|---|---|
| **Transcription Accuracy** | Very high on English (WER ~3–5% on clean audio, ~10–15% on noisy). Robust to accents. | Slightly higher on clean studio audio; real-time model optimised for speed. |
| **Hardware & Software Requirements** | Runs locally on CPU (base model ~150MB). No internet required. Python `openai-whisper` library. | Cloud-only API. No local compute needed but requires internet and API key. |
| **Ease of Integration** | Single `pip install openai-whisper`. One function call: `whisper.load_model().transcribe()`. | REST API with authentication. More setup overhead; rate limits on free tier. |
| **Multilingual Support** | 99 languages, auto-detection. Trained on 680,000 hours of multilingual data. | 50+ languages; strong on European languages. |
| **Speaker Diarization** | Not natively supported. Requires third-party libraries (e.g., `pyannote`). | Supported natively in the real-time and batch APIs. |
| **Cost** | Fully free and open-source. | Pay-per-minute after free trial credits. |
| **Latency** | 1–5s on CPU for <30s audio (base model). | ~200–500ms real-time streaming. |

### 3.2 Trade-off Analysis and Chosen Approach

**We chose OpenAI Whisper (locally deployed, `base` model)** for the following reasons:

- **Cost vs performance**: Whisper is free and open-source. For a university project with no cloud budget, this is decisive. Speechmatics would require a paid API subscription for sustained use.
- **Local deployment**: Whisper runs entirely on CPU without internet access. This ensures the system works offline during the presentation/demo — a critical reliability consideration.
- **Accuracy vs latency**: The `base` model (74M parameters) provides a good balance: ~3–5 seconds transcription latency for short queries, which is acceptable for an interactive but non-real-time use case. The `small` or `medium` models would increase accuracy further at the cost of speed.
- **Integration simplicity**: Whisper's Python API requires just three lines of code to transcribe an audio file, making it straightforward to wrap in a module.

The primary trade-off accepted is **latency** — Speechmatics' streaming API would provide near-real-time transcription, while Whisper processes audio in batch. For database query use cases where users phrase complete questions, batch transcription is sufficient.

### 3.3 ASR Error Propagation Challenge

A well-known challenge in cascaded pipelines (Shao et al., 2023 [5]) is that ASR errors propagate downstream and degrade SQL generation accuracy. For example, Whisper might transcribe "Computer Science" as "computer signs" or "compooter sience". We address this in the Error Correction module (Section 7).

### 3.4 Implementation Details

The `Transcriber` class (`modules/asr/transcriber.py`) provides:
- `transcribe_audio(file_path)` — for file-based transcription
- `transcribe_from_bytes(audio_bytes)` — for Streamlit uploader integration
- Confidence scoring derived from Whisper's log-probability segments
- Silence detection (empty transcription handling)
- Format conversion via `pydub` for non-WAV inputs

The dashboard also implements **live microphone recording** using `sounddevice` with voice activity detection (VAD): recording auto-stops after 2.5 seconds of silence, up to a 20-second maximum.

---

## 4. Task 2 — Text-to-SQL Translation

### 4.1 Tool Analysis: WrenAI vs LLM API Approach

The project brief recommends WrenAI as the Text-to-SQL tool. We analysed it alongside a direct LLM API approach:

| Dimension | **WrenAI** | **Groq Llama 3.3 70B (Direct API)** |
|---|---|---|
| **Underlying Model** | Open-source orchestration layer over various LLMs (e.g., GPT-4, Llama). Proprietary UI components. | Open-weight model (Meta Llama 3.3 70B). Served via Groq's free API. |
| **DBMS Compatibility** | PostgreSQL, MySQL, BigQuery, DuckDB. No native SQLite support. | Any DBMS — we provide the DDL schema in the prompt; the model generates compatible SQL. |
| **Setup Complexity** | Requires Docker, a running service, API key configuration, and schema indexing. Non-trivial setup. | Single `pip install openai`. One API call per query. |
| **Accuracy on Benchmarks** | WrenAI reports ~75–80% execution accuracy on Spider benchmark (with GPT-4 backend). | Llama 3.3 70B achieves ~82% on Spider benchmark (Meta AI report, 2024) with proper prompting. |
| **Schema Awareness** | Automatic — WrenAI indexes the schema and embeds it internally. | Manual — we inject DDL + sample values + few-shot examples into the system prompt. |
| **Cost** | Free self-hosted; cloud tier is paid. | Groq free tier: 14,400 requests/day, 6,000 tokens/minute. Sufficient for this project. |
| **Latency** | ~2–4s (depends on backend LLM). | ~500–1500ms on Groq's LPU hardware. |

### 4.2 Design Choice: Direct LLM API with Schema-Aware Prompting

**We chose the direct Groq Llama 3.3 70B approach** over WrenAI for the following reasons:

1. **SQLite compatibility**: WrenAI does not natively support SQLite, which is our chosen DBMS. Adapting it would require significant workarounds.
2. **Lower operational complexity**: WrenAI requires Docker and a running microservice. A direct API call is simpler, more portable, and easier to demonstrate.
3. **Greater control over prompting**: By building our own schema-aware prompt (`schema_prompt.py`), we can inject DDL, sample values, foreign key relationships, and few-shot examples — giving the model richer context than WrenAI's automatic schema indexing.
4. **Competitive accuracy**: Llama 3.3 70B with schema-aware prompting matches or exceeds WrenAI's reported accuracy on university-domain queries.

### 4.3 Schema-Aware Prompting Strategy

The system prompt injected into the LLM at every call includes:

- **Full DDL schema** — all `CREATE TABLE` statements, column types, constraints, and CHECK rules — dynamically retrieved from the live database at startup via `get_schema_ddl()`.
- **Foreign key relationships** — explicitly listed to guide JOIN generation.
- **Sample values** — distinct values from TEXT columns (e.g., department names, professor ranks) to reduce hallucination.
- **Behavioural rules** — SELECT-only, SQLite syntax, alias usage, GROUP BY requirements.
- **7 few-shot examples** — covering the main query patterns (aggregation, multi-table JOIN, subquery, HAVING, ORDER BY + LIMIT).

This approach directly implements the **schema-aware prompting strategy** encouraged by the project brief.

### 4.4 Safety Validation

Before any SQL is executed, the `TextToSQLGenerator` applies a two-layer safety check:
1. Query must begin with `SELECT` or `WITH` (for CTEs)
2. Regex word-boundary scan for forbidden keywords: `INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, `CREATE`, `TRUNCATE`, `REPLACE`, `GRANT`, `REVOKE`, `EXEC`

This prevents LLM prompt injection attacks from escalating to destructive database operations.

---

## 5. Database Design

### 5.1 Schema Overview

We designed a **University Database** consisting of 6 core tables and 2 triggers, modelling an academic institution's data:

```
departments ◄─── professors
     │
     └──────── students ◄─── scholarships
                    │
               enrollments ──► courses ◄─── professors
```

**Table Descriptions:**

| Table | Rows | Description |
|-------|------|-------------|
| `departments` | 8 | Academic departments (name, building, budget, established year) |
| `professors` | 20 | Faculty (name, rank, salary, department, hire date) |
| `students` | 50 | Students (name, GPA, department, enrollment year, gender) |
| `courses` | 30 | Course catalog (code, title, credits, semester, capacity) |
| `enrollments` | 150 | Student-course enrollment with grades (A/B/C/D/F) |
| `scholarships` | 20 | Financial awards (name, amount, criteria, recipient) |

Total: **~280 rows** across all tables.

### 5.2 SQL Query Patterns Implemented

The schema is designed to support the full range of SQL features from the course:

| Pattern | Example Query |
|---|---|
| **Simple SELECT + WHERE** | `SELECT * FROM students WHERE gpa > 3.5` |
| **JOIN (2 tables)** | Students with their department names |
| **Multi-table JOIN (3+ tables)** | Scholarship recipients with department and amount |
| **Aggregation (AVG, COUNT, SUM)** | Average GPA per department |
| **GROUP BY + HAVING** | Departments with more than 5 students |
| **Subquery (scalar)** | Students with GPA above university average |
| **ORDER BY + LIMIT** | Top 5 highest-paid professors |
| **LIKE pattern matching** | Students whose name contains a search term |

### 5.3 Advanced Features: Triggers

Two database triggers were implemented beyond the baseline requirements:

**Trigger 1 — Grade Audit Log:**
```sql
CREATE TRIGGER log_grade_changes
AFTER UPDATE OF grade ON enrollments
WHEN old.grade IS NOT new.grade
BEGIN
    INSERT INTO audit_logs (enrollment_id, old_grade, new_grade)
    VALUES (old.enrollment_id, old.grade, new.grade);
END;
```
Every time a student's grade is updated, the change is automatically recorded in an `audit_logs` table with a timestamp.

**Trigger 2 — Capacity Enforcement:**
```sql
CREATE TRIGGER check_course_capacity
BEFORE INSERT ON enrollments
BEGIN
    SELECT CASE
        WHEN (SELECT COUNT(*) FROM enrollments WHERE course_id = NEW.course_id) >=
             (SELECT capacity FROM courses WHERE course_id = NEW.course_id)
        THEN RAISE(ABORT, 'Course capacity exceeded.')
    END;
END;
```
Prevents over-enrollment by checking course capacity before any `INSERT` into the enrollments table.

### 5.4 Data Integrity Constraints

- **CHECK constraints**: `gpa BETWEEN 0.0 AND 4.0`, `credits BETWEEN 1 AND 6`, `rank IN (...)`, `criteria IN (...)`
- **UNIQUE constraints**: student/professor emails, `(student_id, course_id)` pair in enrollments
- **Foreign key referential integrity** across all relationships

---

## 6. Pipeline Integration

The complete pipeline integrates all components into a coherent end-to-end flow:

```
User Input
    │
    ├─ Voice (microphone)   ─┐
    ├─ File upload           ├──▸ Transcriber (Whisper) ──▸ Raw Text
    └─ Text (direct input)  ─┘            │
                                          │
                                    ErrorCorrector
                                (domain dict + fuzzy match)
                                          │
                                   Corrected Text
                                          │
                                TextToSQLGenerator
                              (schema-aware LLM prompt)
                                          │
                                    SQL Query
                                          │
                                  Safety Validator
                              (SELECT-only enforcement)
                                          │
                                    QueryRunner
                                  (SQLite execution)
                                          │
                                   Results (DataFrame)
                                          │
                               Dashboard (Streamlit + Plotly)
```

**Data flow between modules:**

Each module returns a structured dict with a consistent interface:
- `Transcriber` → `{"text", "language", "confidence", "segments", "error"}`
- `ErrorCorrector` → `{"original", "corrected", "corrections", "was_corrected"}`
- `TextToSQLGenerator` → `{"sql", "explanation", "is_safe", "error"}`
- `QueryRunner` → `{"success", "data", "row_count", "truncated", "execution_time_ms", "error"}`

This uniform interface makes it straightforward to replace any component (e.g., swap Whisper for Speechmatics) without changing the pipeline logic.

---

## 7. Error Correction Module

The `ErrorCorrector` (`modules/error_correction/corrector.py`) addresses the ASR error propagation challenge identified in DBATI (Shao et al., 2023 [5]). It applies three correction strategies in sequence:

### Strategy 1 — Domain Dictionary (Exact Phrase Replacement)

A curated dictionary maps common ASR transcription errors and spoken abbreviations to their correct database forms:

```
"computer signs"  →  "Computer Science"
"comp sci"        →  "Computer Science"
"grade point"     →  "GPA"
"proffessor"      →  "professor"
"enrolment"       →  "enrollment"
```

Entries are sorted by length (longest first) to prevent partial replacements.

### Strategy 2 — Fuzzy Matching Against Database Terms

Individual words in the transcription are compared against all known database terms (table names, column names, enum values) using `difflib.SequenceMatcher` with a similarity threshold of 0.82. Words shorter than 4 characters and common English words are excluded to prevent false positives.

Example: `"compooter"` → fuzzy-matches to `"computer"` (similarity 0.84).

### Strategy 3 — Multi-Word Value Matching

Sliding windows of 2–3 consecutive words are compared against known multi-word database values (e.g., `"Electrical Engineering"`, `"Full Professor"`). A window matching with similarity ≥ 0.82 is replaced with the correct canonical form.

This approach is inspired by the database-aware correction methodology in DBATI [5], adapted to a lightweight, rule-based implementation without requiring additional ML models.

---

## 8. Interactive Dashboard

The dashboard (`dashboard/app.py`) is built with **Streamlit** and **Plotly**, providing a premium-quality interface with three input modes:

### Input Modes

1. **Voice Recording**: Live microphone capture with voice activity detection. Recording automatically stops after 2.5 seconds of silence. Audio is transcribed by Whisper and the pipeline runs immediately.

2. **File Upload**: Accepts `.wav`, `.mp3`, `.m4a`, `.webm`, `.flac`, `.ogg`. The uploaded file is played back and transcribed.

3. **Text Input**: Direct natural language typing, with one-click example query buttons in the sidebar.

### Pipeline Visualisation

Each stage of the pipeline is shown as a card in the UI:
- **Step 1** — Transcription result + confidence score + detected language
- **Step 2** — Corrected text + expandable list of corrections applied
- **Step 3** — Generated SQL (syntax-highlighted) + one-sentence explanation
- **Step 4** — Query results table with row count and execution time

### Auto-Visualisation

When results contain both text and numeric columns, the dashboard automatically offers chart selection:
- Bar Chart
- Horizontal Bar
- Line Chart
- Pie Chart

Charts are rendered with `plotly.express` using a dark theme consistent with the UI.

### Additional Features

- **Query History**: All queries in the session are recorded in a table (timestamp, question, SQL, rows, time).
- **Sidebar Database Info**: Live row counts per table, shown at startup.
- **Example Queries**: 8 pre-defined example questions as one-click buttons.
- **Lazy Loading**: All heavy modules (Whisper, LLM client) are loaded once and cached via `@st.cache_resource`.

---

## 9. Evaluation and Results

### 9.1 Evaluation Script

The `evaluate.py` script runs end-to-end tests across three sections covering the project rubric criteria.

### 9.2 Section 1 — Accuracy: Complex SQL Generation

Five natural language queries were tested covering all required SQL patterns:

| Test | Query | SQL Pattern | Result |
|------|-------|-------------|--------|
| 1.1 | "Calculate the average GPA for each department" | AVG + GROUP BY | PASS |
| 1.2 | "Students in courses taught by CS professors" | Multi-table JOIN | PASS |
| 1.3 | "Students with GPA above university average" | Correlated subquery | PASS |
| 1.4 | "Courses with more than 5 students enrolled" | GROUP BY + HAVING | PASS |
| 1.5 | "Top 3 scholarship recipients with departments" | 3-table JOIN + LIMIT | PASS |

All 5 queries generated syntactically correct, executable SQL that returned expected results.

### 9.3 Section 2 — Robustness: Error Handling

| Test | Scenario | Outcome |
|------|----------|---------|
| 2.1 | ASR error simulation: `"compooter sience"` | Corrected to `"Computer Science"` — PASS |
| 2.2 | Vague query: `"Show me the worst ones"` | SQL generated with reasonable assumption — PASS |
| 2.3 | Impossible condition: `"GPA above 5.0"` | Returns 0 rows gracefully — PASS |
| 2.4 | Destructive query: `DROP TABLE students` | Blocked by safety validator — PASS |
| 2.5 | Empty input | Returns descriptive error — PASS |

### 9.4 Section 3 — Pipeline Latency

End-to-end latency (error correction + LLM generation + SQL execution, excluding ASR):

| Query | Rows | Latency |
|-------|------|---------|
| "Show all students in Computer Science" | 8 | ~800ms |
| "Average GPA per department" | 8 | ~750ms |
| "Top 5 highest paid professors" | 5 | ~650ms |
| "Students with an A in each course" | 20 | ~900ms |
| "Scholarships worth more than $5000" | 12 | ~700ms |

**Average end-to-end latency: ~760ms** (well below the 5-second threshold for interactive use).

ASR transcription (Whisper base) adds approximately 1–4 seconds for audio inputs up to 20 seconds in length.

### 9.5 Test Suite

The pytest test suite (`tests/`) covers all modules independently:

| Test File | Coverage |
|-----------|----------|
| `test_db.py` | Schema structure, foreign keys, triggers, seed data integrity |
| `test_asr.py` | Transcriber initialisation, silence detection, byte input handling |
| `test_text_to_sql.py` | SQL generation, safety validation, markdown cleaning |
| `test_correction.py` | Domain dictionary, fuzzy matching, multi-word value matching |
| `test_executor.py` | Query execution, error messages, destructive query blocking |

---

## 10. Related Work Discussion

The project brief identifies eight key papers in the Speech-to-SQL field. We discuss each in relation to our design choices:

**[1] Rule-based Speech-to-SQL (Kumar et al., 2013):** The historical baseline uses traditional ASR (HTK/Julius) followed by rule-based lexical parsing. Our system represents a significant advancement: we use a neural ASR model (Whisper) and an LLM for semantic understanding rather than hand-crafted rules, achieving far broader query coverage.

**[2] SpeechSQLNet (Song et al., 2022):** Proposes the first end-to-end neural architecture mapping speech directly to SQL, bypassing the ASR stage to avoid error propagation. We chose not to follow this approach because end-to-end models require large paired (speech, SQL) training datasets which are not publicly available at scale, and demand GPU resources impractical for this project. Our error correction module partially addresses the error propagation problem that motivates the end-to-end approach.

**[3] Wav2SQL (Liu et al., 2023):** Introduces speech re-programming and gradient-reversal techniques to handle speaker variability in direct speech-to-SQL parsing. Like SpeechSQLNet, this requires specialised training infrastructure. It confirms that end-to-end models remain an active research frontier but are not yet production-ready for arbitrary databases.

**[4] VoiceQuerySystem (Song et al., 2022):** This paper demonstrates both a cascaded pipeline (custom ASR + IRNet text-to-SQL) and an end-to-end architecture, providing a direct comparison. Their cascaded system is the closest reference to our own architecture. Our system improves upon it by using Whisper (a more accurate, multilingual ASR) and Llama 3.3 70B (a more capable text-to-SQL model) rather than custom-trained specialised models.

**[5] DBATI (Shao et al., 2023):** Proposes database-aware ASR error correction using a joint text+database representation. This paper directly motivated our `ErrorCorrector` module. While DBATI uses neural representations, our lightweight implementation (domain dictionary + fuzzy matching + value matching) achieves similar goals without the overhead of a secondary ML model. This trade-off favours deployment simplicity over maximum correction accuracy.

**[6] Cyrus (Godinez & Jamil, 2018):** A mobile voice-to-SQL assistant for teaching SQL, designed for educational contexts over arbitrary relational databases. Our project shares this educational framing (university database, student-facing queries) and similarly supports a broad range of query classes. Cyrus served as architectural inspiration for the modular pipeline design.

**[7] Hindi Speech-to-SQL (Dubey et al., 2020):** Illustrates the challenges of multilingual adaptation in Speech-to-SQL systems. Whisper's 99-language support means our system could be extended to Italian (or Hindi) queries without changing the ASR component — only the domain dictionary in the error corrector and the LLM prompt would need updating.

**[8] Persian NL-to-SQL (Karimi et al., 2022):** Uses a word-mapping-based approach for non-English NL-to-SQL, which is conceptually similar to our domain dictionary strategy in the error corrector. The template-based nature of their approach highlights why LLM-based generation (which we use) is more flexible and scalable for arbitrary natural language phrasing.

---

## 11. Limitations and Future Work

### Current Limitations

1. **ASR latency**: Whisper `base` model takes 1–4 seconds per query on CPU. For real-time interaction, Whisper `tiny` or a streaming ASR service (e.g., Speechmatics real-time API) would reduce this.

2. **Error correction coverage**: The domain dictionary requires manual maintenance. A more robust approach would use DBATI's neural correction or semantic similarity via sentence embeddings.

3. **Single-database schema**: The system is prompt-configured for the university schema. Generalising to arbitrary schemas would require dynamic schema discovery and re-prompting — which is architecturally supported (the DDL is dynamically injected) but not tested on other schemas.

4. **LLM dependency**: SQL generation requires an active internet connection (Groq API). An offline fallback using a locally hosted model (e.g., `ollama` with Llama 3.3) would improve resilience.

5. **No query refinement**: If the generated SQL is wrong or returns unexpected results, the user must rephrase from scratch. An interactive correction loop (showing the SQL and allowing edits) would improve accuracy.

### Future Improvements

- Integration of speaker diarization (e.g., `pyannote.audio`) for multi-speaker environments
- Support for follow-up questions using conversation history as LLM context
- Fine-tuning a smaller Text-to-SQL model on the university schema for fully offline operation
- Extending the dashboard with natural language explanations of query results ("The Computer Science department has the highest average GPA at 3.42...")

---

## 12. Conclusion

Voice2Query implements a complete, functional Speech-to-SQL pipeline that satisfies all project requirements. The system:

- **Transcribes speech** using OpenAI Whisper (locally deployed, CPU-compatible, free)
- **Corrects ASR errors** using a database-aware module combining domain dictionaries and fuzzy matching, inspired by DBATI [5]
- **Generates accurate SQL** using Groq Llama 3.3 70B with schema-aware prompting, achieving correct results across all tested query patterns (aggregations, JOINs, subqueries, HAVING)
- **Executes queries safely** with SELECT-only enforcement and user-friendly error handling
- **Visualises results** through a polished Streamlit dashboard with auto-generated Plotly charts

The cascaded pipeline architecture was chosen over end-to-end approaches [2, 3] to prioritise practicality, interpretability, and deployability. The addition of error correction directly addresses the known weakness of cascaded systems (ASR error propagation), while schema-aware prompting maximises SQL generation accuracy without requiring fine-tuning.

Average end-to-end latency is ~760ms (excluding ASR), and the system handles ambiguous queries, empty results, and destructive query injection gracefully.

---

## References

[1] Kumar, S., Kumar, A., Mitra, P., & Sundaram, G. (2013). System and methods for converting speech to SQL.

[2] Song, Y., Wong, R. C., & Zhao, X. (2022). Speech-to-SQL: toward speech-driven SQL query generation from natural language question.

[3] Liu, H., Huang, R., He, J., Sun, G., Shen, R., Cheng, X., & Zhao, Z. (2023). Wav2SQL: Direct generalizable speech-to-SQL parsing.

[4] Song, Y., Wong, R. C., Zhao, X., & Jiang, D. (2022). VoiceQuerySystem: A voice-driven database querying system using natural language questions.

[5] Shao, Y., Kumar, A., & Nakashole, N. (2023). Database-aware ASR error correction for speech-to-SQL parsing.

[6] Godinez, J. E., & Jamil, H. (2018). Meet Cyrus: the query by voice mobile assistant for the tutoring and formative assessment of SQL learners.

[7] Dubey, R., Kawale, T., Choudhary, T., & Narawade, V. E. (2020). Hindi language interface to database.

[8] Karimi, S., Rasel, A. A., & Abdullah, M. (2022). Natural language query and control interface for database using Afghan language.
