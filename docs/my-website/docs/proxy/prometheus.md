import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# 📈 Prometheus metrics [BETA]

LiteLLM Exposes a `/metrics` endpoint for Prometheus to Poll

## Quick Start

If you're using the LiteLLM CLI with `litellm --config proxy_config.yaml` then you need to `pip install prometheus_client==0.20.0`. **This is already pre-installed on the litellm Docker image**

Add this to your proxy config.yaml 
```yaml
model_list:
 - model_name: gpt-3.5-turbo
    litellm_params:
      model: gpt-3.5-turbo
litellm_settings:
  success_callback: ["prometheus"]
  failure_callback: ["prometheus"]
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
| `litellm_requests_metric`             | Number of requests made, per `"user", "key", "model", "team", "end-user"`          |
| `litellm_spend_metric`                | Total Spend, per `"user", "key", "model", "team", "end-user"`                 |
| `litellm_total_tokens`         | input + output tokens per `"user", "key", "model", "team", "end-user"`     |
| `litellm_llm_api_failed_requests_metric`   | Number of failed LLM API requests per `"user", "key", "model", "team", "end-user"`    |

### Budget Metrics
| Metric Name          | Description                          |
|----------------------|--------------------------------------|
| `litellm_remaining_team_budget_metric`             | Remaining Budget for Team (A team created on LiteLLM) |
| `litellm_remaining_api_key_budget_metric`                | Remaining Budget for API Key (A key Created on LiteLLM)|


### ✨ (Enterprise) LLM Remaining Requests and Remaining Tokens
Set this on your config.yaml to allow you to track how close you are to hitting your TPM / RPM limits on each model group 

```yaml
litellm_settings:
  success_callback: ["prometheus"]
  failure_callback: ["prometheus"]
  return_response_headers: true # ensures the LLM API calls track the response headers
```

| Metric Name          | Description                          |
|----------------------|--------------------------------------|
| `litellm_remaining_requests_metric`             | Track `x-ratelimit-remaining-requests` returned from LLM API Deployment |
| `litellm_remaining_tokens`                | Track `x-ratelimit-remaining-tokens` return from LLM API Deployment |

Example Metric
<Tabs>

<TabItem value="Remaining Requests" label="Remaining Requests">

```shell
litellm_remaining_requests
{
  api_base="https://api.openai.com/v1",
  api_provider="openai",
  litellm_model_name="gpt-3.5-turbo",
  model_group="gpt-3.5-turbo"
} 
8998.0
```

</TabItem>

<TabItem value="Requests" label="Remaining Tokens">

```shell
litellm_remaining_tokens
{
  api_base="https://api.openai.com/v1",
  api_provider="openai",
  litellm_model_name="gpt-3.5-turbo",
  model_group="gpt-3.5-turbo"
} 
999981.0
```

</TabItem>

</Tabs>

## Monitor System Health

To monitor the health of litellm adjacent services (redis / postgres), do:

```yaml
model_list:
 - model_name: gpt-3.5-turbo
    litellm_params:
      model: gpt-3.5-turbo
litellm_settings:
  service_callback: ["prometheus_system"]
```

| Metric Name          | Description                          |
|----------------------|--------------------------------------|
| `litellm_redis_latency`         | histogram latency for redis calls     |
| `litellm_redis_fails`         | Number of failed redis calls    |
| `litellm_self_latency`         | Histogram latency for successful litellm api call    |

## 🔥 Community Maintained Grafana Dashboards 

Link to Grafana Dashboards made by LiteLLM community 

https://github.com/BerriAI/litellm/tree/main/cookbook/litellm_proxy_server/grafana_dashboard