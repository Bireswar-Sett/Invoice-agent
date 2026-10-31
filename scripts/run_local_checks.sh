#!/usr/bin/env bash
set -euo pipefail
python -m compileall -q invoice_agent run.py tests scripts/diagnose.py
python -m pytest -q
