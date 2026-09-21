#!/bin/sh

# stale samples from a previous container incarnation would be summed into the aggregate
if [ -n "$PROMETHEUS_MULTIPROC_DIR" ]; then
    mkdir -p "$PROMETHEUS_MULTIPROC_DIR"
    rm -f "$PROMETHEUS_MULTIPROC_DIR"/*.db
fi

case "$USE_DDTRACE" in
    [Tt][Rr][Uu][Ee])
        export DD_TRACE_OPENAI_ENABLED="False"
        exec ddtrace-run "$@"
        ;;
esac

exec "$@"
