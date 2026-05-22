#!/usr/bin/env bash
# Build napparent_tabular (maturin) and run the CSV smoke test.
# Usage:
#   ./scripts/run_napparent_tabular_bindings.sh
#   ./scripts/run_napparent_tabular_bindings.sh --release
#   ./scripts/run_napparent_tabular_bindings.sh --skip-build -- --limit 2000
#
# Extra arguments after -- are passed to test_napparent_tabular_bindings.py.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TEST_SCRIPT="$SCRIPT_DIR/test_napparent_tabular_bindings.py"

RELEASE=0
SKIP_BUILD=0
PY_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --release)
      RELEASE=1
      shift
      ;;
    --skip-build)
      SKIP_BUILD=1
      shift
      ;;
    --)
      shift
      PY_ARGS+=("$@")
      break
      ;;
    *)
      PY_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
  PYTHON="${VIRTUAL_ENV}/bin/python"
elif [[ -n "${NAPPARENT_TABULAR_PYTHON:-}" && -x "$NAPPARENT_TABULAR_PYTHON" ]]; then
  PYTHON="$NAPPARENT_TABULAR_PYTHON"
elif [[ -n "${PYTHON:-}" && -x "$PYTHON" ]]; then
  PYTHON="$PYTHON"
elif [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
  PYTHON="$REPO_ROOT/.venv/bin/python"
else
  PYTHON="python3"
fi

py_major="$("$PYTHON" -c 'import sys; print(sys.version_info[0])')"
py_minor="$("$PYTHON" -c 'import sys; print(sys.version_info[1])')"
if [[ "$py_major" -lt 3 ]] || [[ "$py_major" -eq 3 && "$py_minor" -lt 10 ]]; then
  echo "napparent-tabular requires Python 3.10 or newer." >&2
  exit 1
fi

if [[ ! -f "$TEST_SCRIPT" ]]; then
  echo "Missing $TEST_SCRIPT" >&2
  exit 1
fi

if [[ "$SKIP_BUILD" -eq 0 ]]; then
  cd "$REPO_ROOT"
  if [[ "$RELEASE" -eq 1 ]]; then
    env PYO3_PYTHON="$PYTHON" maturin develop --release
  else
    env PYO3_PYTHON="$PYTHON" maturin develop
  fi
fi

if [[ "${#PY_ARGS[@]}" -eq 0 ]]; then
  exec "$PYTHON" "$TEST_SCRIPT"
else
  exec "$PYTHON" "$TEST_SCRIPT" "${PY_ARGS[@]}"
fi
