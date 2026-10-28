#!/bin/sh

if [ "$1" = "--admin-agent" ]; then
    shift
    export CONNECTION_AUTH_MODE=native
    exec /opt/liteadmin/bin/litellm-admin-agent --web "$@"
fi

case "$USE_DDTRACE" in
    [Tt][Rr][Uu][Ee])
        export DD_TRACE_OPENAI_ENABLED="False"
        exec ddtrace-run litellm "$@"
        ;;
esac

exec litellm "$@"
