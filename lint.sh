#!/usr/bin/env bash
# Runs all linters: black (formatting), ruff (lint + import sorting), pyrefly (type checking).
# Usage: ./lint.sh          check only
#        ./lint.sh --fix    apply black formatting and safe ruff fixes first, then check

set -uo pipefail
cd "$(dirname "$0")"

if [[ "${1:-}" == "--fix" ]]; then
    poetry run ruff check --fix --quiet .
    poetry run black --quiet .
fi

status=0
run() {
    echo "=== $*"
    "$@" || status=1
}

run poetry run black --check .
run poetry run ruff check .
run poetry run pyrefly check

if [[ $status -eq 0 ]]; then
    echo "All checks passed."
else
    echo "Some checks failed."
fi
exit $status
