#!/bin/sh
# Hosted-runner wrapper: run one benchmark leg, then serve its output over HTTP.
#
# Render one-off jobs don't surface stdout via the Logs API, so on a hosted
# runner we instead make the long-lived web service RUN the leg and PUBLISH the
# result file. The single argument is a base64-encoded, space-separated wsbench
# flag list (same encoding as run.sh — dodges Render's start-command tokenizer).
#
# Flow: decode flags → run wsbench, tee output to /tmp/web/result.txt → exec
# busybox httpd serving /tmp/web on $PORT. Fetch the result at
# http://<runner>/result.txt. No secrets are written: the flags (incl. -key) are
# only passed to wsbench's argv, never echoed into the served file.
set -e
ARGS_B64="$1"
DECODED="$(echo "$ARGS_B64" | base64 -d)"
mkdir -p /tmp/web

# Run the leg; capture stdout (the SUMMARY + per-connection breakdown) to the
# served file. wsbench prints no secrets. `|| true` so httpd still comes up to
# serve whatever was captured even if the run errors.
# shellcheck disable=SC2086 -- intentional word-splitting of the flag list
/usr/local/bin/wsbench $DECODED > /tmp/web/result.txt 2>&1 || true
echo "__WSBENCH_DONE__" >> /tmp/web/result.txt

exec httpd -f -p "${PORT:-10000}" -h /tmp/web
