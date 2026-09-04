"""
Database Setup Script
=====================
Creates the SQLite database, applies the schema, and inserts mock data.
Idempotent: drops and recreates all tables on each run.

Usage:
    python database/setup_db.py
"""

import sqlite3
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from config import DB_PATH, SCHEMA_SQL_PATH, SEED_SQL_PATH


def setup_database() -> None:
    """Create the database, apply schema, and insert seed data."""
    print(f"📦 Setting up database at: {DB_PATH}")

    # Read SQL files
    schema_sql = SCHEMA_SQL_PATH.read_text(encoding="utf-8")
    seed_sql = SEED_SQL_PATH.read_text(encoding="utf-8")

    # Connect and execute
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()

    try:
        # Enable foreign keys
        cursor.execute("PRAGMA foreign_keys = ON;")

        # Apply schema (includes DROP IF EXISTS)
        print("🔧 Applying schema...")
        cursor.executescript(schema_sql)

        # Insert mock data
        print("🌱 Inserting seed data...")
        cursor.executescript(seed_sql)

        conn.commit()

        # Verify tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
        tables = [row[0] for row in cursor.fetchall()]
        print(f"\n✅ Created {len(tables)} tables: {', '.join(tables)}")

        # Print row counts
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table};")
            count = cursor.fetchone()[0]
            print(f"   📊 {table}: {count} rows")

        print(f"\n🎉 Database setup complete! File: {DB_PATH}")

        # =====================================================================
        # ADVANCED FEATURE: TRIGGER DEMONSTRATION
        # =====================================================================
        print("\n" + "="*50)
        print("  ADVANCED SQL TRIGGERS DEMONSTRATION")
        print("="*50)

        # 1. Test the Capacity Trigger
        # Let's find a course and temporarily lower its capacity to 2 for the test
        cursor.execute("SELECT course_id, capacity FROM courses LIMIT 1;")
        course_id, original_capacity = cursor.fetchone()
        
        # Use a temporary capacity of 2 — save existing enrollments first so we can restore them
        cursor.execute("SELECT enrollment_id, student_id, grade, enrollment_date FROM enrollments WHERE course_id = ?;", (course_id,))
        saved_enrollments = cursor.fetchall()

        cursor.execute("UPDATE courses SET capacity = 2 WHERE course_id = ?;", (course_id,))
        print(f"🔧 Testing Capacity Trigger on course {course_id} (capacity lowered to 2)...")

        # Remove existing enrollments for this course to start fresh for the test
        cursor.execute("DELETE FROM enrollments WHERE course_id = ?;", (course_id,))

        # Fill it up to capacity (2 students)
        test_student_ids = []
        for i in range(2):
            cursor.execute("SELECT student_id FROM students WHERE student_id NOT IN (SELECT student_id FROM enrollments WHERE course_id = ?) LIMIT 1;", (course_id,))
            student_id = cursor.fetchone()[0]
            cursor.execute("INSERT INTO enrollments (student_id, course_id, enrollment_date) VALUES (?, ?, '2026-05-09');", (student_id, course_id))
            test_student_ids.append(student_id)

        # Now try to exceed capacity (Trigger should block this)
        try:
            cursor.execute("SELECT student_id FROM students WHERE student_id NOT IN (SELECT student_id FROM enrollments WHERE course_id = ?) LIMIT 1;", (course_id,))
            student_id = cursor.fetchone()[0]
            cursor.execute("INSERT INTO enrollments (student_id, course_id, enrollment_date) VALUES (?, ?, '2026-05-09');", (student_id, course_id))
            print("❌ FAIL: Trigger did not block the over-enrollment.")
        except sqlite3.IntegrityError as e:
            if "Course capacity exceeded" in str(e):
                print("✅ PASS: BEFORE INSERT Trigger successfully blocked over-enrollment!")
            else:
                print(f"❌ FAIL: Unexpected error: {e}")

        # Restore original capacity and seeded enrollments
        cursor.execute("UPDATE courses SET capacity = ? WHERE course_id = ?;", (original_capacity, course_id))
        cursor.execute("DELETE FROM enrollments WHERE course_id = ?;", (course_id,))
        for eid, sid, grade, edate in saved_enrollments:
            cursor.execute(
                "INSERT INTO enrollments (enrollment_id, student_id, course_id, grade, enrollment_date) VALUES (?, ?, ?, ?, ?);",
                (eid, sid, course_id, grade, edate)
            )

        # 2. Test the Audit Log Trigger
        print(f"\n🔧 Testing Audit Log Trigger on enrollments...")
        
        # Get an enrollment to change
        cursor.execute("SELECT enrollment_id, grade FROM enrollments LIMIT 1;")
        row = cursor.fetchone()
        if row:
            enrollment_id, old_grade = row
            new_grade = 'A+' if old_grade != 'A+' else 'B'
            
            # Update the grade
            cursor.execute("UPDATE enrollments SET grade = ? WHERE enrollment_id = ?;", (new_grade, enrollment_id))
            
            # Verify the trigger populated the audit_logs table
            cursor.execute("SELECT old_grade, new_grade FROM audit_logs WHERE enrollment_id = ? ORDER BY changed_at DESC LIMIT 1;", (enrollment_id,))
            log = cursor.fetchone()
            
            if log and log[0] == old_grade and log[1] == new_grade:
                print("✅ PASS: AFTER UPDATE Trigger successfully recorded grade change in audit_logs!")
                print(f"   Logged change: {old_grade} → {new_grade}")
            else:
                print("❌ FAIL: Audit log was not created correctly.")

        conn.commit()

    except Exception as e:
        conn.rollback()
        print(f"❌ Error setting up database: {e}")
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    setup_database()
