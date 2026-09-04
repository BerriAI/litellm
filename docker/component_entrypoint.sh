#!/bin/sh

case "$USE_DDTRACE" in
    [Tt][Rr][Uu][Ee])
        export DD_TRACE_OPENAI_ENABLED="False"
        exec ddtrace-run "$@"
        ;;
esac

exec "$@"
