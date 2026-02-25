#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ ! -f "main.py" ]]; then
  echo "[error] main.py not found in: $ROOT_DIR"
  exit 1
fi

if [[ -z "${WSL_DISTRO_NAME:-}" ]]; then
  echo "[warn] WSL environment variable not found. Continue anyway."
fi

PY_BIN="${PYTHON_BIN:-}"
if [[ -z "$PY_BIN" ]]; then
  if command -v python3.10 >/dev/null 2>&1; then
    PY_BIN="python3.10"
  elif command -v python3 >/dev/null 2>&1; then
    PY_BIN="python3"
  else
    echo "[error] python3 not found. Please install Python 3.10+ in WSL."
    exit 1
  fi
fi

VENV_DIR="${VENV_DIR:-.venv-wsl}"
if [[ ! -d "$VENV_DIR" ]]; then
  echo "[setup] creating virtualenv: $VENV_DIR"
  "$PY_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

if [[ "${SKIP_INSTALL:-0}" != "1" ]]; then
  echo "[setup] installing/updating Python packages..."
  python -m pip install -U pip setuptools wheel
  python -m pip install -U \
    "tensorflow==2.20.0" \
    numpy \
    pyyaml \
    matplotlib \
    tqdm \
    optuna \
    plotly \
    requests
else
  echo "[setup] SKIP_INSTALL=1, skip package install."
fi

echo "[check] validating TensorFlow GPU visibility..."
python - <<'PY'
import sys
import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
print(f"[check] TensorFlow: {tf.__version__}")
print(f"[check] GPU devices: {gpus}")

if not gpus:
    print("[error] TensorFlow cannot see a GPU in WSL.")
    print("[hint] Update NVIDIA Windows driver, run `wsl --update`, then reboot.")
    sys.exit(2)
PY

echo "[run] python main.py $*"
python main.py "$@"
