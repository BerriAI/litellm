# LiteLLM All Prometheus Metrics dashboard

Every `litellm_*` metric family the proxy can expose on `/metrics` (134 families across 95 panels), grouped into rows: proxy traffic, latency, spend and tokens, cache, LLM API deployments, key and team rate limits, budgets, guardrails, MCP, managed files and batches, users and teams, the Redis circuit breaker, the spend log cleanup job, and the `prometheus_system` service callback metrics (per-service latency, request and failure rates, spend update queue sizes). Panel titles are the metric names so you can grep the JSON for the metric you care about

Import `grafana_dashboard.json` from **Dashboards > New > Import** and pick your Prometheus data source when prompted (the `DS_PROMETHEUS` variable). Counters are plotted as `rate()` over `$__rate_interval`, histograms as p50 / p95 / p99, gauges as the raw value grouped by the most useful label. Every query names the metric exactly as the proxy emits it (counters carry the `_total` suffix the Prometheus client adds), and `tests/test_litellm/integrations/test_prometheus_metric_name_consistency.py` fails if a metric is renamed without updating this dashboard

The first eleven rows need only `callbacks: ["prometheus"]`. The last three rows and the `litellm_admission_*` panels are emitted by other subsystems and stay empty until those are on: the service callback row needs `service_callback: ["prometheus_system"]` in `litellm_settings`, the circuit breaker row needs a Redis cache, the cleanup row needs spend log retention, and admission control needs its middleware enabled. Within the base rows, many panels only fill in once the matching feature is in use: budgets need keys, teams, users or orgs with `max_budget` set, cache panels need caching on, guardrail and MCP panels need those features configured, deployment health needs the router with more than one deployment or a failure to record, and `litellm_in_flight_requests` needs traffic at scrape time. An empty panel for a feature you do not use is expected

## Pre-requisites

Prometheus metrics on the proxy: https://docs.litellm.ai/docs/proxy/prometheus
