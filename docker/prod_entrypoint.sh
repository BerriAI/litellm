#!/bin/sh

# Azure Monitor OTEL auto-instrumentation. Picked up by opentelemetry-instrument
# automatically when azure-monitor-opentelemetry distro is installed and
# APPLICATIONINSIGHTS_CONNECTION_STRING is set. Set OTEL_DISABLED=true to bypass.
if [ -n "${APPLICATIONINSIGHTS_CONNECTION_STRING}" ] && [ "${OTEL_DISABLED:-false}" != "true" ]; then
    OTEL_WRAP="opentelemetry-instrument"
else
    OTEL_WRAP=""
fi

if [ "$USE_DDTRACE" = "true" ]; then
    export DD_TRACE_OPENAI_ENABLED="False"
    exec ${OTEL_WRAP} ddtrace-run litellm "$@"
else
    exec ${OTEL_WRAP} litellm "$@"
fi
