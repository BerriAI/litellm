# LiteLLM GenAI dashboard (OpenTelemetry metrics)

Dashboard for the `gen_ai.*` metrics the OpenTelemetry v2 integration emits, as opposed to the `litellm_*` Prometheus metrics the other dashboards in this folder chart.

Import `grafana_dashboard.json` from **Dashboards > New > Import** and pick your Prometheus data source. Panels: request count, spend, token count, p95 duration, request rate by model, spend rate per hour by model, tokens per minute split by input and output, p95 duration by model, p95 time to first token, and p95 provider generation time. Template variables for data source, service, and model.

## Pre-requisites

Metrics are off by default. In the proxy environment:

```shell
LITELLM_OTEL_V2=true
LITELLM_OTEL_INTEGRATION_ENABLE_METRICS=true
OTEL_EXPORTER="otlp_http"
OTEL_ENDPOINT="<your OTLP endpoint>"
```

You also need the metric attribute filter, or the panels will plot flat lines at zero. LiteLLM's default attribute set includes per-request fields, so nearly every request lands in its own time series with a single sample, and `rate()` has nothing to compute over:

```yaml title="config.yaml"
callback_settings:
  otel:
    attributes:
      include_list:
        - gen_ai.operation.name
        - gen_ai.system
        - gen_ai.request.model
        - gen_ai.framework
```

See [Grafana Cloud](https://docs.litellm.ai/docs/observability/grafana_cloud) for the full setup, and [OpenTelemetry v2](https://docs.litellm.ai/docs/observability/opentelemetry_v2#metrics) for the metric reference.

## Note on Grafana's AI Observability integration

Grafana Cloud ships prebuilt GenAI dashboards that query these same metric names, so they look like a drop-in alternative to this one. They are not: twenty of their twenty-two panels filter on `telemetry_sdk_name="openlit"`, a label LiteLLM does not carry and cannot be configured to add, so those panels stay empty.
