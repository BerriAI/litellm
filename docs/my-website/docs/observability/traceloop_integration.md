# Traceloop (via OpenTelemetry)

Traceloop is configured through LiteLLM’s OpenTelemetry integration.

## Set up

```python
litellm.callbacks = ["otel"]  # enable OpenTelemetry logger
```

```shell
OTEL_EXPORTER="otlp_http"
OTEL_ENDPOINT="https://api.traceloop.com"
OTEL_HEADERS="Authorization=Bearer%20<your-api-key>"
```

Set `OTEL_EXPORTER`, `OTEL_ENDPOINT`, and `OTEL_HEADERS` in the environment before starting the process.

For SDK usage details and callback usage across providers, refer to the shared [OpenTelemetry integration guide](../observability/opentelemetry_integration.md).
