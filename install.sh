#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "Creating project virtual environment: ${VENV_DIR}"
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

VENV_PYTHON="${VENV_DIR}/bin/python"
if [[ ! -x "${VENV_PYTHON}" ]]; then
  echo "ERROR: ${VENV_PYTHON} does not exist or is not executable." >&2
  exit 1
fi

echo "Using Python: ${VENV_PYTHON}"
"${VENV_PYTHON}" -m pip install --upgrade pip
"${VENV_PYTHON}" -m pip install --no-user -e "${ROOT_DIR}[all]"

VENV_PROSPECTOR="${VENV_DIR}/bin/prospector"
if [[ ! -x "${VENV_PROSPECTOR}" ]]; then
  echo "ERROR: Prospector was not installed into ${VENV_DIR}/bin/." >&2
  echo "Refusing to continue because the installation target is wrong." >&2
  exit 1
fi

echo
echo "Prospector installed into the project virtual environment."
echo "Executable: ${VENV_DIR}/bin/prospector"
echo
echo "Activate it with:"
echo "  source ${VENV_DIR}/bin/activate"
echo "Then run:"
echo "  prospector --version"
