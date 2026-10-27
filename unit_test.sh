#!/usr/bin/env bash
# Runs the unit tests of the latentspace package. Extra arguments are passed to unittest (e.g., -v or -k pattern).
# Usage: ./unit_test.sh [unittest options]

set -euo pipefail
# the tests package is discovered relative to the repository root
cd "$(dirname "$0")"

poetry run python -m unittest discover -b "$@" tests
