# Silent model comparison observability failure

## Status

Observed on 2026-07-24 in a LiteLLM deployment using a primary model and a background `silent_model`

This document records the observed failure modes and current observability gaps. It does not propose or implement a fix

## Test shape

Each client request is expected to produce two provider calls:

1. The primary model call returned to the client
2. A background silent-model call used only for comparison

The examined workload used:

- Primary LiteLLM model: `bailian/deepseek-v4-flash`
- Silent LiteLLM model: `glm-5.2`
- Silent provider model: `zai-org/GLM-5.2-FP8`
- Shared key alias: `llm-shadow-yj-pre-20260723`

## Observed symptoms

### Request-log counts do not appear one-to-one

The LiteLLM Request Logs UI showed different totals when filtering by deployment:

- GLM deployment: 134 rows
- DeepSeek deployment: 125 rows

Querying the same time range by provider-model name returned 137 raw rows for each model. The apparent discrepancy is caused by several different record shapes being combined in the UI totals:

- DeepSeek had failures that occurred before a deployment `model_id` was attached, so a UI filter for a specific deployment excluded those rows
- The screenshots and the API query used slightly different end times
- Seven GLM session IDs each had three GLM records, producing 14 additional GLM rows
- Cache-hit records sometimes used different session IDs for the primary and silent sides

Raw row equality therefore does not prove that every primary request has exactly one corresponding silent request

### Session-level pairing is incomplete

For the inspected interval:

| Measurement | GLM | DeepSeek |
| --- | ---: | ---: |
| Raw rows | 137 | 137 |
| Unique session IDs | 123 | 137 |
| Success rows | 127 | 111 |
| Cache-hit rows | 10 | 5 |
| Failure rows | 0 | 21 |

Only 111 session IDs formed clean, successful pairs. There were 12 GLM-only session IDs and 26 DeepSeek-only session IDs

The DeepSeek failures included `APIConnectionError`, `BudgetExceededError`, `ProxyRateLimitError`, and `TypeError`. Some failures did not contain a deployment ID. Cache hits and duplicate calls also prevented a strict one-row-per-role join by session ID

### The silent model has a large latency tail

For the 111 clean pairs, the two calls started within a few milliseconds of each other, confirming that they belonged to the same primary/silent comparison attempt

| Duration | GLM silent | DeepSeek primary |
| --- | ---: | ---: |
| P50 | 15.0 s | 3.95 s |
| P90 | 80.1 s | 9.74 s |
| P95 | 112.0 s | 11.8 s |
| P99 | 154.4 s | 21.4 s |
| Maximum | 346.2 s | 27.0 s |

The per-session GLM-to-DeepSeek duration ratio had a median of 3.45x, a P95 of 22.96x, and a maximum of 62.38x

GLM frequently generated more output tokens. The paired output-token ratio had a median of 1.62x and a P90 of 5.69x. Output length explains part of the latency difference, but not all of it: one pair took approximately 100 seconds on GLM for 422 output tokens and 3.7 seconds on DeepSeek for 181 output tokens

### Existing Prometheus metrics cannot identify bad pairs

The deployed `litellm_request_total_latency_metric` exposes aggregate histogram labels such as:

- `requested_model`
- `model`
- `model_id`
- `api_provider`
- key, team, organization, and user dimensions

It does not expose `session_id`, a stable shadow-pair identifier, or a primary/silent request role

As a result, Prometheus can compare aggregate model latency distributions, but it cannot answer the following questions from existing metrics:

- Which primary and silent samples belong to the same request
- Which session IDs have the largest duration ratio or duration delta
- Which primary requests succeeded while their silent counterparts failed or disappeared
- Whether count differences came from missing pairs, retries, cache hits, or duplicate calls

Adding `session_id` as a normal Prometheus label would create one time series per request and is not a safe workaround for this gap. Histogram metrics also retain bucket counts rather than individual observations, so a per-session join cannot be reconstructed after collection

### TTFT is not comparable between the two roles

The silent call is forced to use `stream=False` so that the background response is fully consumed and callbacks execute. The primary request may use streaming

For the non-streaming silent request, a displayed TTFT can be equal or close to the full request duration. It must not be compared directly with streaming primary-model TTFT

### OpenTelemetry does not currently preserve the pair as one trace

OpenTelemetry is a suitable data model for this comparison because a primary model span and a silent model span could be represented under one request trace, with duration, status, token usage, and request role stored as span attributes

The current silent-experiment implementation does not provide that structure automatically:

- The silent request runs in a background thread with a new event loop
- `_get_silent_experiment_kwargs` removes `metadata.litellm_parent_otel_span` before starting the silent request because a live span object cannot safely cross the event-loop boundary
- The silent call receives a fresh LiteLLM call ID and logging context
- The current OpenTelemetry GenAI span attributes do not expose the general LiteLLM `session_id` as a first-class comparison attribute
- No span link is created between the primary span and the silent span

Enabling an OTLP exporter alone will therefore produce useful individual LLM spans, but it will not reliably make the existing primary and silent calls queryable as one trace. The trace relationship or an explicit shadow-pair correlation must first be propagated across the background boundary

## User impact

Operators can see that one model has a worse aggregate P95 or P99, but they cannot quickly move from that signal to a ranked list of the affected paired requests

The current investigation path requires exporting or querying Spend Logs, joining rows by session ID, removing cache hits and duplicates, and calculating duration ratios manually. This makes performance regressions, missing silent requests, and provider-specific failures slower to detect and diagnose

## Reproduction and validation criteria

The observability failure is present when all of the following are true:

1. A request is configured with a `silent_model`
2. Request Logs contain primary and silent rows that can sometimes be correlated by session ID
3. Prometheus exposes separate aggregate latency series for both models
4. No existing metric query or trace query can return a ranked, exact list of paired requests by duration ratio or duration delta
5. Cache hits, failures, or duplicate calls cause UI row counts or unique-session counts to diverge

Any future fix should be validated against successful pairs, primary-only and silent-only records, provider failures without a deployment ID, cache hits, retries, duplicate calls, streaming primary calls, and non-streaming silent calls
