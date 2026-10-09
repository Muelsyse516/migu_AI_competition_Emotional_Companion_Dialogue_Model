#!/bin/bash
set -euo pipefail
if [ "$#" -ne 2 ]; then echo "Usage: start.sh TEST_FILE RESULT_DIR" >&2; exit 2; fi
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTHONUTF8=1
PYTHON_BIN="${PARTICIPANT_PYTHON:-/opt/conda/envs/conda_env/bin/python3}"
if [ ! -x "$PYTHON_BIN" ]; then
  if [ -n "${PARTICIPANT_PYTHON:-}" ]; then
    echo "RUNTIME_ERROR: explicitly selected interpreter missing: $PYTHON_BIN" >&2
    exit 127
  fi
  PYTHON_BIN="$(command -v python3 || true)"
  if [ -z "$PYTHON_BIN" ]; then echo "RUNTIME_ERROR: no Python interpreter available" >&2; exit 127; fi
  echo "Conda path absent; validating available interpreter: $PYTHON_BIN" >&2
fi
export PYTHONPATH="$ROOT_DIR/runtime_vendor${PYTHONPATH:+:$PYTHONPATH}"
LOG_ROOT="${TMPDIR:-/tmp}"
LOG_FILE="$(mktemp "$LOG_ROOT/participant-v5.XXXXXX.log")"
echo "Diagnostic log: $LOG_FILE" >&2
set +e
(
  "$PYTHON_BIN" "$ROOT_DIR/preflight.py" &&
  "$PYTHON_BIN" "$ROOT_DIR/inference.py" "$1" "$2"
) 2>&1 | tee "$LOG_FILE"
STATUS=${PIPESTATUS[0]}
set -e
echo "PARTICIPANT_EXIT_CODE=$STATUS LOG=$LOG_FILE" >&2
exit "$STATUS"
