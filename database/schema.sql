-- Voice2Query — University Database Schema
-- Supports: JOINs, aggregations, subqueries, GROUP BY + HAVING

DROP TABLE IF EXISTS scholarships;
DROP TABLE IF EXISTS enrollments;
DROP TABLE IF EXISTS courses;
DROP TABLE IF EXISTS professors;
DROP TABLE IF EXISTS students;
DROP TABLE IF EXISTS departments;

CREATE TABLE departments (
    department_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT NOT NULL UNIQUE,
    building        TEXT NOT NULL,
    budget          INTEGER NOT NULL,
    established     INTEGER NOT NULL
);

CREATE TABLE professors (
    professor_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    first_name      TEXT NOT NULL,
    last_name       TEXT NOT NULL,
    email           TEXT NOT NULL UNIQUE,
    rank            TEXT NOT NULL CHECK (rank IN ('Assistant Professor','Associate Professor','Full Professor','Lecturer')),
    salary          REAL NOT NULL,
    department_id   INTEGER NOT NULL,
    hire_date       TEXT NOT NULL,
    FOREIGN KEY (department_id) REFERENCES departments(department_id)
);

CREATE TABLE students (
    student_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    first_name      TEXT NOT NULL,
    last_name       TEXT NOT NULL,
    email           TEXT NOT NULL UNIQUE,
    date_of_birth   TEXT NOT NULL,
    gender          TEXT NOT NULL CHECK (gender IN ('Male','Female','Non-binary')),
    department_id   INTEGER NOT NULL,
    enrollment_year INTEGER NOT NULL,
    gpa             REAL NOT NULL CHECK (gpa >= 0.0 AND gpa <= 4.0),
    FOREIGN KEY (department_id) REFERENCES departments(department_id)
);

CREATE TABLE courses (
    course_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    course_code     TEXT NOT NULL UNIQUE,
    title           TEXT NOT NULL,
    credits         INTEGER NOT NULL CHECK (credits BETWEEN 1 AND 6),
    department_id   INTEGER NOT NULL,
    professor_id    INTEGER NOT NULL,
    semester        TEXT NOT NULL,
    capacity        INTEGER NOT NULL,
    FOREIGN KEY (department_id) REFERENCES departments(department_id),
    FOREIGN KEY (professor_id) REFERENCES professors(professor_id)
);

CREATE TABLE enrollments (
    enrollment_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id      INTEGER NOT NULL,
    course_id       INTEGER NOT NULL,
    grade           TEXT,
    enrollment_date TEXT NOT NULL,
    FOREIGN KEY (student_id) REFERENCES students(student_id),
    FOREIGN KEY (course_id) REFERENCES courses(course_id),
    UNIQUE(student_id, course_id)
);

CREATE TABLE scholarships (
    scholarship_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT NOT NULL,
    amount          REAL NOT NULL,
    student_id      INTEGER NOT NULL,
    awarded_date    TEXT NOT NULL,
    criteria        TEXT NOT NULL CHECK (criteria IN ('Merit','Need-based','Athletic','Research','Diversity')),
    FOREIGN KEY (student_id) REFERENCES students(student_id)
);

-- ============================================================================
-- ADVANCED FEATURES: TRIGGERS & AUDIT LOGS
-- ============================================================================

-- 1. Audit Table for Grade Changes
DROP TABLE IF EXISTS audit_logs;
CREATE TABLE audit_logs (
    log_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    enrollment_id   INTEGER NOT NULL,
    old_grade       TEXT,
    new_grade       TEXT,
    changed_at      DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- 2. Trigger: Automatically log grade changes
CREATE TRIGGER log_grade_changes
AFTER UPDATE OF grade ON enrollments
WHEN old.grade IS NOT new.grade
BEGIN
    INSERT INTO audit_logs (enrollment_id, old_grade, new_grade)
    VALUES (old.enrollment_id, old.grade, new.grade);
END;

-- 3. Trigger: Enforce Course Capacity Constraint
CREATE TRIGGER check_course_capacity
BEFORE INSERT ON enrollments
BEGIN
    SELECT CASE
        WHEN (SELECT COUNT(*) FROM enrollments WHERE course_id = NEW.course_id) >= 
             (SELECT capacity FROM courses WHERE course_id = NEW.course_id)
        THEN RAISE(ABORT, 'Course capacity exceeded. Cannot enroll student.')
    END;
END;

