#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
fail() { printf 'Startup error: %s\n' "$*" >&2; exit 2; }
PORT="${MARIMO_PORT:-8080}"
[[ "$PORT" =~ ^[0-9]{1,5}$ ]] || fail 'MARIMO_PORT must be an integer from 1 to 65535'
PORT="$((10#$PORT))"
(( PORT >= 1 && PORT <= 65535 )) || fail 'MARIMO_PORT must be from 1 to 65535'
export MARIMO_PORT="$PORT"
MODE="${MARIMO_MODE:-edit}"
[[ "$MODE" == edit || "$MODE" == run ]] || fail 'MARIMO_MODE must be edit or run'
ROOT="${NOTEBOOK_ROOT:-/workspace}"
[[ "$ROOT" = /* ]] || fail 'NOTEBOOK_ROOT must be an absolute path'
mkdir -p -- "$ROOT"
[[ -w "$ROOT" ]] || fail 'NOTEBOOK_ROOT must be writable'

export CACHE_ROOT="${CACHE_ROOT:-/workspace/.cache}"
export HF_HOME="${HF_HOME:-$CACHE_ROOT/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$CACHE_ROOT/torch}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$CACHE_ROOT/uv}"
mkdir -p -- "$HF_HOME" "$TORCH_HOME" "$UV_CACHE_DIR"

ARGS=("$MODE" --headless --host "${MARIMO_HOST:-0.0.0.0}" --port "$PORT" --token --no-sandbox)
[[ "$MODE" != edit ]] || ARGS+=(--skip-update-check)
# Keep passwords out of argv. With neither setting, marimo generates a token.
if [[ -n "${MARIMO_TOKEN_PASSWORD:-}" && -n "${MARIMO_TOKEN_PASSWORD_FILE:-}" ]]; then
  fail 'Set only one of MARIMO_TOKEN_PASSWORD and MARIMO_TOKEN_PASSWORD_FILE'
elif [[ -n "${MARIMO_TOKEN_PASSWORD_FILE:-}" ]]; then
  [[ -s "$MARIMO_TOKEN_PASSWORD_FILE" && -r "$MARIMO_TOKEN_PASSWORD_FILE" ]] || fail 'Token file must be readable and nonempty'
  ARGS+=(--token-password-file "$MARIMO_TOKEN_PASSWORD_FILE")
elif [[ -n "${MARIMO_TOKEN_PASSWORD:-}" ]]; then
  ARGS+=(--token-password-file -)
fi

PUBLIC="${PUBLIC_HOST:-}"
if [[ -z "$PUBLIC" && -n "${RUNPOD_POD_ID:-}" ]]; then
  PUBLIC="${RUNPOD_POD_ID}-${PORT}.proxy.runpod.net:443"
fi
[[ -z "$PUBLIC" ]] || ARGS+=(--proxy "$PUBLIC")
[[ -z "${MARIMO_BASE_URL:-}" ]] || ARGS+=(--base-url "$MARIMO_BASE_URL")
# marimo accepts repeated flags, not a single comma-separated origin string.
if [[ -n "${ALLOW_ORIGINS:-}" ]]; then
  IFS=',' read -ra ORIGINS <<< "$ALLOW_ORIGINS"
  for origin in "${ORIGINS[@]}"; do
    [[ -z "$origin" ]] || ARGS+=(--allow-origins "$origin")
  done
fi
if [[ "$MODE" == run ]]; then
  [[ "${NOTEBOOK_FILE:-}" = /* && -f "${NOTEBOOK_FILE:-}" && "$NOTEBOOK_FILE" == *.py ]] || fail 'run mode requires NOTEBOOK_FILE pointing to an existing absolute .py path'
  TARGET="$NOTEBOOK_FILE"
else
  TARGET="$ROOT"
fi

case "${SEED_EXAMPLES:-0}" in
  1) mkdir -p -- "$ROOT/examples"
     # Copy only diagnostics, never overwrite users' notebooks.
     if [[ ! -e "$ROOT/examples/hardware_check.py" ]]; then
       cp -n -- "$APP_DIR/examples/hardware_check.py" "$ROOT/examples/hardware_check.py"
     fi ;;
  0) ;;
  *) fail 'SEED_EXAMPLES must be 0 or 1' ;;
esac

python "$APP_DIR/scripts/hardware_check.py"
cd -- "$ROOT"
if [[ -n "${MARIMO_TOKEN_PASSWORD:-}" ]]; then
  PASSWORD="$MARIMO_TOKEN_PASSWORD"
  unset MARIMO_TOKEN_PASSWORD
  exec marimo "${ARGS[@]}" "$TARGET" <<< "$PASSWORD"
else
  exec marimo "${ARGS[@]}" "$TARGET"
fi
