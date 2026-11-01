#!/usr/bin/env bash
# Reference solution: shows the task can be solved, and that the hidden tests accept a correct answer.
set -euo pipefail
python3 -c "
from pathlib import Path
export = Path('export.py')
export.write_text(export.read_text().replace('%m/%d/%Y', '%Y-%m-%d'))
"
