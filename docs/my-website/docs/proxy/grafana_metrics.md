# Grafana, Prometheus metrics [BETA]

LiteLLM Exposes a `/metrics` endpoint for Prometheus to Poll

## Quick Start

Add this to your proxy config.yaml 
```yaml
model_list:
 - model_name: gpt-3.5-turbo
    litellm_params:
      model: gpt-3.5-turbo
litellm_settings:
  success_callback: ["prometheus"]
```

Start the proxy
```shell
litellm --config config.yaml --debug
```

Test Request
```shell
curl --location 'http://0.0.0.0:4000/chat/completions' \
    --header 'Content-Type: application/json' \
    --data '{
    "model": "gpt-3.5-turbo",
    "messages": [
        {
        "role": "user",
        "content": "what llm are you"
        }
    ]
}'
```

View Metrics on `/metrics`, Visit `http://localhost:4000/metrics` 
```shell
http://localhost:4000/metrics

# <proxy_base_url>/metrics
```

## Metrics Tracked 


| Metric Name          | Description                          |
|----------------------|--------------------------------------|
| `litellm_requests_metric`             | Number of requests made, per `"user", "key", "model"`          |
| `litellm_spend_metric`                | Total Spend, per `"user", "key", "model"`                 |
| `litellm_total_tokens`         | input + output tokens per `"user", "key", "model"`     |
