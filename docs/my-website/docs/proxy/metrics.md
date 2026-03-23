# Prometheus Metrics Configuration

## Custom Latency Buckets

By default, LiteLLM uses a fixed set of histogram buckets for all latency metrics. You can override them to match your SLOs.

```yaml
litellm_settings:
  success_callback: ["prometheus"]
  prometheus_latency_buckets: [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, .inf]
```

Applies to all histogram metrics:
- `litellm_request_total_latency_metric`
- `litellm_llm_api_latency_metric`
- `litellm_llm_api_time_to_first_token_metric`
- `litellm_overhead_latency_metric`
- `litellm_request_queue_time_seconds`
- `litellm_guardrail_latency_seconds`

---

## Exclude Metrics

Disable specific metrics entirely — they are replaced by no-ops and never registered in Prometheus.

```yaml
litellm_settings:
  success_callback: ["prometheus"]
  prometheus_exclude_metrics:
    - litellm_overhead_latency_metric
    - litellm_request_queue_time_seconds
```

---

## Exclude Labels

Strip label dimensions from **all** metrics. Useful for reducing cardinality.

```yaml
litellm_settings:
  success_callback: ["prometheus"]
  prometheus_exclude_labels:
    - end_user
    - user_agent
    - client_ip
```

> `prometheus_exclude_labels` is applied after any `prometheus_metrics_config` include-label filtering.

---

# 💸 GET Daily Spend, Usage Metrics

## Request Format
```shell
curl -X GET "http://0.0.0.0:4000/daily_metrics" -H "Authorization: Bearer sk-1234"
```

## Response format
```json
[
    daily_spend = [
        {
            "daily_spend": 7.9261938052047e+16,
            "day": "2024-02-01T00:00:00",
            "spend_per_model": {"azure/gpt-4": 7.9261938052047e+16},
            "spend_per_api_key": {
                "76": 914495704992000.0,
                "12": 905726697912000.0,
                "71": 866312628003000.0,
                "28": 865461799332000.0,
                "13": 859151538396000.0
            }
        },
        {
            "daily_spend": 7.938489251309491e+16,
            "day": "2024-02-02T00:00:00",
            "spend_per_model": {"gpt-3.5": 7.938489251309491e+16},
            "spend_per_api_key": {
                "91": 896805036036000.0,
                "78": 889692646082000.0,
                "49": 885386687861000.0,
                "28": 873869890984000.0,
                "56": 867398637692000.0
            }
        }

    ],
    total_spend = 200,
    top_models = {"gpt4": 0.2, "vertexai/gemini-pro":10},
    top_api_keys = {"899922": 0.9, "838hcjd999seerr88": 20}

]

```
