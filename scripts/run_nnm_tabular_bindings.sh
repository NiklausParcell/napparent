#!/usr/bin/env bash
# Build nnm_tabular (maturin) and run the CSV smoke test.
# Usage:
#   ./scripts/run_nnm_tabular_bindings.sh
#   ./scripts/run_nnm_tabular_bindings.sh --release
#   ./scripts/run_nnm_tabular_bindings.sh --skip-build -- --limit 2000
#
# Extra arguments after -- are passed to test_nnm_tabular_bindings.py.
# Without --, any argument that is not --release / --skip-build is passed to Python.
#
# Interpreter resolution (for the test script and PYO3_PYTHON): uses
# $VIRTUAL_ENV/bin/python if set, else NNM_TABULAR_PYTHON, else $PYTHON,
# else rust-learn/.venv/bin/python, else python3.
#
# Activate a Python 3.10+ venv before running so maturin develop installs
# deps into that environment (maturin develop has no --python option).
#
# Requires Python 3.10+ (same as nnm-tabular-rust/pyproject.toml); older versions
# cannot install pyarrow>=15.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
NNM_DIR="$REPO_ROOT/nnm-tabular-rust"
TEST_SCRIPT="$SCRIPT_DIR/test_nnm_tabular_bindings.py"

# Avoid "${array[@]}" on empty arrays with `set -u` (breaks on macOS /bin/bash 3.2).
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
elif [[ -n "${NNM_TABULAR_PYTHON:-}" && -x "$NNM_TABULAR_PYTHON" ]]; then
  PYTHON="$NNM_TABULAR_PYTHON"
elif [[ -n "${PYTHON:-}" && -x "$PYTHON" ]]; then
  PYTHON="$PYTHON"
elif [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
  PYTHON="$REPO_ROOT/.venv/bin/python"
else
  PYTHON="python3"
fi

# Keep in sync with nnm-tabular-rust/pyproject.toml requires-python
py_major="$("$PYTHON" -c 'import sys; print(sys.version_info[0])')"
py_minor="$("$PYTHON" -c 'import sys; print(sys.version_info[1])')"
if [[ "$py_major" -lt 3 ]] || [[ "$py_major" -eq 3 && "$py_minor" -lt 10 ]]; then
  echo "nnm-tabular requires Python 3.10 or newer; \`pyarrow>=15\` has no builds for older Pythons." >&2
  echo "This interpreter is: $("$PYTHON" --version 2>&1) ($PYTHON)" >&2
  echo "Recreate the venv, for example:" >&2
  echo "  cd \"$REPO_ROOT\" && rm -rf .venv && python3.12 -m venv .venv && source .venv/bin/activate && pip install -U pip maturin pyarrow" >&2
  exit 1
fi

if [[ ! -f "$TEST_SCRIPT" ]]; then
  echo "Missing $TEST_SCRIPT" >&2
  exit 1
fi

if [[ "$SKIP_BUILD" -eq 0 ]]; then
  cd "$NNM_DIR"
  # maturin develop attaches to the active venv; PYO3_PYTHON aligns the PyO3 build with $PYTHON.
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
