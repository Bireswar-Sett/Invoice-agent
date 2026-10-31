#!/usr/bin/env bash
set -euo pipefail

# Requires Render CLI and an authenticated session: render login
# Set RENDER_SERVICE_ID to avoid selecting the service interactively.
SERVICE_ID="${RENDER_SERVICE_ID:-}"
OUT_DIR="${RENDER_LOG_DIR:-./diagnostics}"
mkdir -p "$OUT_DIR"

if ! command -v render >/dev/null 2>&1; then
  echo "Render CLI is not installed. See https://render.com/docs/cli" >&2
  exit 1
fi

if [[ -z "$SERVICE_ID" ]]; then
  echo "Find the service ID with: render services"
  echo "Then run: RENDER_SERVICE_ID=srv-... $0"
  exit 2
fi

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
echo "Collecting recent Render logs for $SERVICE_ID (UTC $STAMP)"

# All recent runtime logs, including Python stack traces.
render logs --resources "$SERVICE_ID" --limit 100 --output json \
  > "$OUT_DIR/render-all-$STAMP.json"

# Error-level logs only.
render logs --resources "$SERVICE_ID" --level error --limit 100 --output json \
  > "$OUT_DIR/render-errors-$STAMP.json"

# Application logs containing the message handler's diagnostic marker.
render logs --resources "$SERVICE_ID" --text "Unhandled message processing failure" --limit 100 --output json \
  > "$OUT_DIR/render-message-errors-$STAMP.json" || true

# Request-log filtering is available where the workspace plan exposes request logs.
render logs --resources "$SERVICE_ID" --method POST --path /a2a/message:send --status-code 502 --limit 100 --output json \
  > "$OUT_DIR/render-a2a-502-$STAMP.json" || true

printf '\nSaved logs in %s\n' "$OUT_DIR"
printf 'Before sharing, inspect files for secrets, invoice content, or bearer tokens.\n'
