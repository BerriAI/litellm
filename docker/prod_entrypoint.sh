#!/bin/sh

if [ "$USE_DDTRACE" = "true" ]; then
    exec ddtrace-run litellm "$@"
else
    exec py-spy litellm "$@"
fi
