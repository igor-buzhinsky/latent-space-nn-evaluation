#!/usr/bin/env bash
# Runs the unit tests of the latentspace package. Extra arguments are passed to unittest (e.g., -v or -k pattern).
# Usage: ./test.sh [unittest options]

set -euo pipefail
# bundled models are loaded by paths relative to the repository root
cd "$(dirname "$0")"

poetry run python -m unittest discover -b "$@" tests
