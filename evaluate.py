"""
Voice2Query — Committee Evaluation Script
==========================================
End-to-end evaluation of the pipeline against the project rubric.
Covers: Accuracy, Robustness, Pipeline Integration.

Usage:
    python evaluate.py
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    from modules.text_to_sql.generator import TextToSQLGenerator
    from modules.error_correction.corrector import ErrorCorrector
    from modules.executor.query_runner import QueryRunner

    generator = TextToSQLGenerator()
    corrector = ErrorCorrector()
    runner = QueryRunner()

    PASS = "✅ PASS"
    FAIL = "❌ FAIL"
    results = []

    def run_test(test_id, description, query):
        """Run a full pipeline test: NL → Correction → SQL → Execute."""
        print(f"\n{'='*70}")
        print(f"  {test_id}: {description}")
        print(f"  Query: \"{query}\"")
        print(f"{'='*70}")

        correction = corrector.correct(query)
        corrected = correction["corrected"]
        if correction["was_corrected"]:
            print(f"  🔧 Corrected: \"{corrected}\"")
            for c in correction["corrections"]:
                print(f"     - {c['from']} → {c['to']} ({c['type']})")
        else:
            print(f"  🔧 No correction needed")

        start = time.time()
        sql_result = generator.generate_sql(corrected)
        gen_time = (time.time() - start) * 1000

        if sql_result["error"]:
            print(f"  ❌ SQL Error: {sql_result['error']}")
            results.append((test_id, description, FAIL, sql_result["error"]))
            return None

        print(f"  🤖 SQL ({gen_time:.0f}ms): {sql_result['sql']}")
        print(f"  💬 Explanation: {sql_result['explanation']}")

        if not sql_result["is_safe"]:
            results.append((test_id, description, FAIL, "Unsafe SQL"))
            return None

        exec_result = runner.execute(sql_result["sql"])
        exec_time = (time.time() - start) * 1000

        if exec_result["success"]:
            print(f"  📊 Results: {exec_result['row_count']} rows in {exec_time:.1f}ms")
            print(exec_result["data"].to_string(index=False, max_rows=5))
            results.append((test_id, description, PASS, f"{exec_result['row_count']} rows"))
        else:
            print(f"  ❌ Execution Error: {exec_result['error']}")
            results.append((test_id, description, FAIL, exec_result["error"]))

        return exec_result

    # ═══════════════════════════════════════════════════════════════════════
    # 1. ACCURACY: Complex SQL Generation
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "█" * 70)
    print("  SECTION 1: ACCURACY — Complex SQL Generation")
    print("█" * 70)

    run_test("1.1", "Aggregations & Grouping (AVG + GROUP BY)",
             "Calculate the average GPA for each department")
    run_test("1.2", "Joins (students + departments)",
             "Show me all students enrolled in courses taught by professors in the Computer Science department")
    run_test("1.3", "Subqueries",
             "Find the students who have a GPA greater than the overall university average")
    run_test("1.4", "GROUP BY + HAVING + COUNT",
             "Which courses have more than 5 students enrolled?")
    run_test("1.5", "Complex multi-table JOIN",
             "List the top 3 scholarship recipients with their department names and amounts")

    # ═══════════════════════════════════════════════════════════════════════
    # 2. ROBUSTNESS: Error Handling and Edge Cases
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "█" * 70)
    print("  SECTION 2: ROBUSTNESS — Error Handling & Edge Cases")
    print("█" * 70)

    # Test 2.1: ASR-like misspelling
    print(f"\n{'='*70}")
    print(f"  2.1: Domain-specific vocabulary / ASR error simulation")
    print(f"{'='*70}")
    misspelled = "Show students in the compooter sience department"
    correction = corrector.correct(misspelled)
    print(f"  Input:     \"{misspelled}\"")
    print(f"  Corrected: \"{correction['corrected']}\"")
    if correction["was_corrected"]:
        for c in correction["corrections"]:
            print(f"     - {c['from']} → {c['to']} ({c['type']})")
        results.append(("2.1", "ASR error correction", PASS, "Corrections applied"))
    else:
        results.append(("2.1", "ASR error correction", FAIL, "No correction made"))

    run_test("2.2", "Ambiguous/vague query", "Show me the worst ones")
    run_test("2.3", "Empty result set (impossible condition)",
             "Show me students with a GPA above 5.0")

    # Test 2.4: Destructive query safety
    print(f"\n{'='*70}")
    print(f"  2.4: Safety — destructive query rejection")
    print(f"{'='*70}")
    exec_result = runner.execute("DROP TABLE students")
    if not exec_result["success"]:
        print(f"  🔒 Blocked: {exec_result['error']}")
        results.append(("2.4", "Destructive query rejection", PASS, "Blocked"))
    else:
        results.append(("2.4", "Destructive query rejection", FAIL, "NOT BLOCKED!"))

    # Test 2.5: Empty input
    sql_result = generator.generate_sql("")
    if sql_result["error"]:
        print(f"\n  2.5: Empty input → Handled: {sql_result['error']}")
        results.append(("2.5", "Empty input handling", PASS, "Error returned"))
    else:
        results.append(("2.5", "Empty input handling", FAIL, "No error"))

    # ═══════════════════════════════════════════════════════════════════════
    # 3. PIPELINE INTEGRATION & LATENCY
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "█" * 70)
    print("  SECTION 3: PIPELINE INTEGRATION — End-to-End Latency")
    print("█" * 70)

    queries = [
        "Show all students in Computer Science",
        "What is the average GPA per department?",
        "Top 5 highest paid professors",
        "How many students got an A in each course?",
        "Find students with scholarships worth more than 5000 dollars",
    ]

    total_latencies = []
    for i, q in enumerate(queries, 1):
        start = time.time()
        corr = corrector.correct(q)
        sql = generator.generate_sql(corr["corrected"])
        latency = (time.time() - start) * 1000
        total_latencies.append(latency)
        if sql.get("sql") and sql.get("is_safe"):
            result = runner.execute(sql["sql"])
            status = PASS if result["success"] else FAIL
            print(f"  3.{i}: \"{q}\" → {result.get('row_count', 0)} rows in {latency:.0f}ms {status}")
        else:
            print(f"  3.{i}: \"{q}\" → SQL Error in {latency:.0f}ms {FAIL}")

    avg_latency = sum(total_latencies) / len(total_latencies)
    print(f"\n  📊 Average end-to-end latency: {avg_latency:.0f}ms")
    results.append(("3.0", f"Avg latency ({avg_latency:.0f}ms)",
                    PASS if avg_latency < 5000 else FAIL, f"{avg_latency:.0f}ms"))

    # ═══════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "█" * 70)
    print("  EVALUATION SUMMARY")
    print("█" * 70)

    passed = sum(1 for r in results if PASS in r[2])
    failed = sum(1 for r in results if FAIL in r[2])

    print(f"\n  {'Test ID':<8} {'Description':<45} {'Status':<10} {'Detail'}")
    print(f"  {'─'*8} {'─'*45} {'─'*10} {'─'*30}")
    for test_id, desc, status, detail in results:
        print(f"  {test_id:<8} {desc:<45} {status:<10} {detail}")

    print(f"\n  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"  Total: {passed + failed} tests | {PASS}: {passed} | {FAIL}: {failed}")
    print(f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


if __name__ == "__main__":
    main()
