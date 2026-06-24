#!/bin/sh
# Wrapper to dodge the Render start-command tokenizer (which lumps args after
# the ~5th token). The single argument is a base64-encoded, space-separated
# flag list. We decode it and exec wsbench with the resulting args.
set -e
ARGS_B64="$1"
DECODED="$(echo "$ARGS_B64" | base64 -d)"
# shellcheck disable=SC2086 -- intentional word-splitting of the flag list
exec /usr/local/bin/wsbench $DECODED
