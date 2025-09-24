#!/usr/bin/env bash
set -euo pipefail

PORT="${MARIMO_PORT:-8080}"
HOST="0.0.0.0"
ROOT="${NOTEBOOK_ROOT:-/workspace}"

# Build the RunPod proxy host if the platform exposes the Pod ID
# Proxy URL format is: https://[pod-id]-[port].proxy.runpod.net/  (TLS via 443)
PROXY_FLAG=()
if [[ -n "${RUNPOD_POD_ID:-}" ]]; then
  PROXY_FLAG=(--proxy "${RUNPOD_POD_ID}-${PORT}.proxy.runpod.net:443")
fi

# Auth: disable token for easier health checks, or set MARIMO_TOKEN_PASSWORD for a fixed token
AUTH_FLAG=()
if [[ -n "${MARIMO_TOKEN_PASSWORD:-}" ]]; then
  AUTH_FLAG=(--token-password "${MARIMO_TOKEN_PASSWORD}")
else
  AUTH_FLAG=(--no-token)
fi

# Optional CORS
ALLOW_ORIGINS="${ALLOW_ORIGINS:-*}"

exec marimo edit \
  --headless \
  --host "${HOST}" \
  --port "${PORT}" \
  --allow-origins "${ALLOW_ORIGINS}" \
  "${AUTH_FLAG[@]}" \
  "${PROXY_FLAG[@]}" \
  "${ROOT}"
