"""
Schema-Aware Prompt Builder
============================
Dynamically constructs LLM prompts that include the database schema,
sample data, and few-shot examples for accurate Text-to-SQL generation.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from database.connection import get_schema_ddl, get_schema_info


def build_system_prompt() -> str:
    """
    Build the system prompt for the Text-to-SQL LLM.
    Includes DDL schema, table relationships, and behavioral instructions.

    Returns:
        str: Complete system prompt with schema context.
    """
    ddl = get_schema_ddl()
    schema_info = get_schema_info()

    # Build relationship summary
    relationships = []
    for table_name, info in schema_info["tables"].items():
        for fk in info["foreign_keys"]:
            relationships.append(f"  - {table_name}.{fk['column']} → {fk['references']}")

    relationships_text = "\n".join(relationships) if relationships else "  (none)"

    # Build sample values summary for key columns
    sample_lines = []
    for table_name, info in schema_info["tables"].items():
        for col_name, values in info.get("sample_values", {}).items():
            if len(values) <= 15:
                sample_lines.append(f"  - {table_name}.{col_name}: {', '.join(values[:10])}")

    samples_text = "\n".join(sample_lines) if sample_lines else "  (none)"

    return f"""You are an expert SQL query generator for a SQLite university database.
Your job is to convert natural language questions into valid, executable SQL queries.

## DATABASE SCHEMA (DDL)

{ddl}

## TABLE RELATIONSHIPS (Foreign Keys)

{relationships_text}

## SAMPLE VALUES (for reference)

{samples_text}

## RULES

1. Generate ONLY SELECT queries. Never generate INSERT, UPDATE, DELETE, DROP, ALTER, or CREATE statements.
2. Use proper JOIN syntax when queries involve multiple tables.
3. Use table aliases for readability (e.g., s for students, d for departments).
4. Always qualify column names with table aliases when joining multiple tables.
5. Use SQLite-compatible syntax only (e.g., no LIMIT with OFFSET without ORDER BY).
6. For aggregations, always include appropriate GROUP BY clauses.
7. Return ONLY the SQL query — no explanations, no markdown, no code fences.
8. If the question is ambiguous, make reasonable assumptions and generate the most likely intended query.
9. For name searches, use LIKE with wildcards for flexibility.
10. When asked about "top" or "best", use ORDER BY with LIMIT.

## FEW-SHOT EXAMPLES

User: "Show me all students in the Computer Science department"
SQL: SELECT s.student_id, s.first_name, s.last_name, s.email, s.gpa FROM students s JOIN departments d ON s.department_id = d.department_id WHERE d.name = 'Computer Science'

User: "What is the average GPA per department?"
SQL: SELECT d.name AS department, ROUND(AVG(s.gpa), 2) AS average_gpa FROM students s JOIN departments d ON s.department_id = d.department_id GROUP BY d.name ORDER BY average_gpa DESC

User: "Find students with GPA above the university average"
SQL: SELECT s.first_name, s.last_name, s.gpa, d.name AS department FROM students s JOIN departments d ON s.department_id = d.department_id WHERE s.gpa > (SELECT AVG(gpa) FROM students) ORDER BY s.gpa DESC

User: "Which departments have more than 5 students?"
SQL: SELECT d.name AS department, COUNT(s.student_id) AS student_count FROM departments d JOIN students s ON d.department_id = s.department_id GROUP BY d.name HAVING COUNT(s.student_id) > 5 ORDER BY student_count DESC

User: "Show the top 5 highest paid professors and their departments"
SQL: SELECT p.first_name, p.last_name, p.salary, p.rank, d.name AS department FROM professors p JOIN departments d ON p.department_id = d.department_id ORDER BY p.salary DESC LIMIT 5

User: "How many students got an A in each course?"
SQL: SELECT c.course_code, c.title, COUNT(e.enrollment_id) AS a_count FROM courses c JOIN enrollments e ON c.course_id = e.course_id WHERE e.grade = 'A' GROUP BY c.course_id, c.course_code, c.title ORDER BY a_count DESC

User: "List all scholarships awarded to Computer Science students"
SQL: SELECT sc.name AS scholarship, sc.amount, sc.criteria, s.first_name, s.last_name FROM scholarships sc JOIN students s ON sc.student_id = s.student_id JOIN departments d ON s.department_id = d.department_id WHERE d.name = 'Computer Science' ORDER BY sc.amount DESC
"""


def build_user_prompt(natural_language_query: str) -> str:
    """
    Build the user message for the LLM.

    Args:
        natural_language_query: The natural language question from the user.

    Returns:
        str: Formatted user prompt.
    """
    return f"Convert this question to SQL: {natural_language_query}"
