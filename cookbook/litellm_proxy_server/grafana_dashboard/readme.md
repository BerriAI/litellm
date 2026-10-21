# Contains LiteLLM maintained grafana dashboard

This folder contains the `json` for creating Grafana Dashboards

## [LiteLLM GenAI Dashboard (OpenTelemetry)](./dashboard_genai_otel)

Charts the `gen_ai.*` metrics from the OpenTelemetry v2 integration: spend, tokens, request rate, and latency percentiles by model. Separate from the dashboards below, which chart the `litellm_*` Prometheus metrics.

## [LiteLLM All Prometheus Metrics dashboard](./dashboard_all_metrics)

Every `litellm_*` Prometheus metric family the proxy can emit (134 families, 95 panels) grouped by theme: traffic, latency, spend and tokens, cache, deployments, rate limits, budgets, guardrails, MCP, managed files and batches, users and teams, plus the Redis circuit breaker, spend log cleanup and `prometheus_system` service metrics. Start here if you want everything on one screen; see its [readme](./dashboard_all_metrics/readme.md) for import steps and which panels need a feature enabled before they show data

## [LiteLLM v2 Dashboard](./dashboard_v2)

A compact view of proxy request rate, failures, latency and the top remaining-request / remaining-token gauges per model group

<img width="1316" alt="grafana_1" src="https://github.com/user-attachments/assets/d0df802d-0cb9-4906-a679-941c547789ab">
<img width="1289" alt="grafana_2" src="https://github.com/user-attachments/assets/b11f755f-e113-42ab-b21d-83f91f451a28">
<img width="1323" alt="grafana_3" src="https://github.com/user-attachments/assets/cb29ffdb-477d-4be1-a5cd-c3f7f2cb21c5">



### Pre-Requisites
- Setup LiteLLM Proxy Prometheus Metrics https://docs.litellm.ai/docs/proxy/prometheus 
