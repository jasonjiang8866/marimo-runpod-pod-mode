#!/usr/bin/env bash
set -euo pipefail

# Where we created the venv in the Dockerfile (uv venv runs in /workspace)
WORKDIR="/workspace"
VENV_DIR="${WORKDIR}/.venv"
PORT="${MARIMO_PORT:-8080}"
HOST="0.0.0.0"

cd "$WORKDIR"

# Activate the uv-managed venv (created by `uv venv`)
if [[ -d "$VENV_DIR" ]]; then
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
else
  echo "ERROR: Virtual environment not found at ${VENV_DIR}."
  echo "Did the Docker build run 'uv venv' successfully?"
  exit 1
fi

# Optional: create a notebooks directory to keep things tidy
NOTEBOOK_ROOT="${NOTEBOOK_ROOT:-$WORKDIR}"
mkdir -p "$NOTEBOOK_ROOT"

# Launch Marimo editor (headless) bound to all interfaces
# Use exec so Marimo becomes PID 1 (proper signal handling in containers)
exec marimo edit --headless --host "${HOST}" --port "${PORT}" "${NOTEBOOK_ROOT}"
