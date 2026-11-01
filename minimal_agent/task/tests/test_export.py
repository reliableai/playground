"""Hidden tests for the ISO-dates task. The agent never sees this folder.

Usage:  python test_export.py <folder>
Runs export.py in <folder> and compares its output with expected.csv, the answer fixed in advance.
Exit code 0 if every test passes, 1 if any fails.
"""
import csv
import io
import subprocess
import sys
from pathlib import Path

EXPECTED = Path(__file__).parent / "expected.csv"
DATE_COLUMNS = [2, 3]
OTHER_COLUMNS = [0, 1, 4]


def read_rows(text):
    return list(csv.reader(io.StringIO(text)))


def columns(rows, indexes):
    """The chosen columns of every row; None for a row without the expected 5 columns."""
    return [[row[index] for index in indexes] if len(row) == 5 else None for row in rows]


def main(folder):
    export = subprocess.run([sys.executable, "export.py"], cwd=folder, capture_output=True, text=True, timeout=30)
    if export.returncode != 0:
        print(f"export.py failed:\n{export.stderr}")
    actual = read_rows(export.stdout) if export.returncode == 0 else []
    expected = read_rows(EXPECTED.read_text())
    results = {
        "test_row_count": len(actual) == len(expected),
        "test_dates_are_iso": columns(actual, DATE_COLUMNS) == columns(expected, DATE_COLUMNS),
        "test_other_columns_unchanged": columns(actual, OTHER_COLUMNS) == columns(expected, OTHER_COLUMNS),
    }
    for name, passed in results.items():
        print(f"  {'PASS' if passed else 'FAIL'} {name}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python test_export.py <folder>")
    sys.exit(main(sys.argv[1]))
