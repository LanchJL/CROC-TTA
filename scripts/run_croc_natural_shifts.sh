#!/usr/bin/env bash
set -euo pipefail

python scripts/run_croc_benchmarks.py --benchmark natural "$@"
