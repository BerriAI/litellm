import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# 📈 [BETA] Prometheus metrics

:::info
🚨 Prometheus metrics will be out of Beta on September 15, 2024 - as part of this release it will be on LiteLLM Enterprise starting at $250/mo

[Enterprise Pricing](https://www.litellm.ai/#pricing)

[Contact us here to get a free trial](https://calendly.com/d/4mp-gd3-k5k/litellm-1-1-onboarding-chat)

:::

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

## 📈 Metrics Tracked 

### Virtual Keys, Teams, Internal Users Metrics

Use this for for tracking per [user, key, team, etc.](virtual_keys)

| Metric Name          | Description                          |
|----------------------|--------------------------------------|
| `litellm_requests_metric`             | Number of requests made, per `"user", "key", "model", "team", "end-user"`          |
| `litellm_spend_metric`                | Total Spend, per `"user", "key", "model", "team", "end-user"`                 |
| `litellm_total_tokens`         | input + output tokens per `"user", "key", "model", "team", "end-user"`     |



### LLM API / Provider Metrics

Use this for LLM API Error monitoring and tracking remaining rate limits and token limits

| Metric Name          | Description                          |
|----------------------|--------------------------------------|
 `litellm_deployment_success_responses`              |  Total number of successful LLM API calls for deployment                               |
| `litellm_deployment_failure_responses`              | Total number of failed LLM API calls for a specific LLM deploymeny. exception_status is the status of the exception from the llm api                                   |
| `litellm_deployment_total_requests`                 | Total number of LLM API calls for deployment - success + failure                      |
| `litellm_remaining_requests_metric`             | Track `x-ratelimit-remaining-requests` returned from LLM API Deployment |
| `litellm_remaining_tokens`                | Track `x-ratelimit-remaining-tokens` return from LLM API Deployment |
| `litellm_deployment_state`             | The state of the deployment: 0 = healthy, 1 = partial outage, 2 = complete outage. |
| `litellm_deployment_latency_per_output_token`       | Latency per output token for deployment                                                          |

### Load Balancing, Fallback, Cooldown Metrics

Use this for tracking [litellm router](../routing) load balancing metrics

| Metric Name          | Description                          |
|----------------------|--------------------------------------|
| `litellm_deployment_cooled_down`             |  Number of times a deployment has been cooled down by LiteLLM load balancing logic. exception_status is the status of the exception that caused the deployment to be cooled down |
| `litellm_deployment_successful_fallbacks`           |  Number of successful fallback requests from primary model -> fallback model        |
| `litellm_deployment_failed_fallbacks`               | Number of failed fallback requests from primary model -> fallback model            |


### Request Latency Metrics 

| Metric Name          | Description                          |
|----------------------|--------------------------------------|
| `litellm_request_total_latency_metric`             | Total latency (seconds) for a request to LiteLLM Proxy Server - tracked for labels `litellm_call_id`, `model` |
| `litellm_llm_api_latency_metric`             | latency (seconds) for just the LLM API call - tracked for labels `litellm_call_id`, `model` |


### Budget Metrics
| Metric Name          | Description                          |
|----------------------|--------------------------------------|
| `litellm_remaining_team_budget_metric`             | Remaining Budget for Team (A team created on LiteLLM) |
| `litellm_remaining_api_key_budget_metric`                | Remaining Budget for API Key (A key Created on LiteLLM)|



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