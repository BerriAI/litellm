#!/bin/sh

if [ "$USE_DDTRACE" = "true" ]; then
    export DD_TRACE_OPENAI_ENABLED="False"
    exec ddtrace-run litellm "$@"

elif [ "$USE_MEMRAY" = "true" ]; then
    # install memray if it’s not already present
    pip install --no-cache-dir memray

    # start the FastAPI/LiteLLM proxy in the background
    litellm "$@" &
    LITELLM_PID=$!

    # attach memray to capture all native and Python heap allocs  
    memray attach \
      --native \
      --output /tmp/memray.bin \
      $LITELLM_PID

    # when memray finishes (or the process exits), wait on the server
    wait $LITELLM_PID

else
    exec litellm "$@"
fi
