import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Provider Budget Routing
Use this to set budgets for LLM Providers - example $100/day for OpenAI, $100/day for Azure.

## Quick Start

Set provider budgets in your `proxy_config.yaml` file
### Proxy Config setup
```yaml
model_list:
    - model_name: gpt-3.5-turbo
      litellm_params:
        model: openai/gpt-3.5-turbo
        api_key: os.environ/OPENAI_API_KEY

router_settings:
  provider_budget_config: 
    openai: 
      budget_limit: 0.000000000001 # float of $ value budget for time period
      time_period: 1d # can be 1d, 2d, 30d, 1mo, 2mo
    azure:
      budget_limit: 100
      time_period: 1d
    anthropic:
      budget_limit: 100
      time_period: 10d
    vertex_ai:
      budget_limit: 100
      time_period: 12d
    gemini:
      budget_limit: 100
      time_period: 12d
  
  # OPTIONAL: Set Redis Host, Port, and Password if using multiple instance of LiteLLM
  redis_host: os.environ/REDIS_HOST
  redis_port: os.environ/REDIS_PORT
  redis_password: os.environ/REDIS_PASSWORD

general_settings:
  master_key: sk-1234
```

### Make a test request

We expect the first request to succeed, and the second request to fail since we cross the budget for `openai`


**[Langchain, OpenAI SDK Usage Examples](../proxy/user_keys#request-format)**

<Tabs>
<TabItem label="Successful Call " value = "allowed">

```shell
curl -i http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{
    "model": "gpt-4o",
    "messages": [
      {"role": "user", "content": "hi my name is test request"}
    ]
  }'
```

</TabItem>
<TabItem label="Unsuccessful call" value = "not-allowed">

Expect this to fail since since `ishaan@berri.ai` in the request is PII

```shell
curl -i http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{
    "model": "gpt-4o",
    "messages": [
      {"role": "user", "content": "hi my name is test request"}
    ]
  }'
```

Expected response on failure

```json
{
  "error": {
    "message": "No deployments available - crossed budget for provider: Exceeded budget for provider openai: 0.0007350000000000001 >= 1e-12",
    "type": "None",
    "param": "None",
    "code": "429"
  }
}
```

</TabItem>


</Tabs>



## How provider budget routing works

1. **Budget Tracking**: 
   - Uses Redis to track spend for each provider
   - Tracks spend over specified time periods (e.g., "1d", "30d")
   - Automatically resets spend after time period expires

2. **Routing Logic**:
   - Routes requests to providers under their budget limits
   - Skips providers that have exceeded their budget
   - If all providers exceed budget, raises an error

3. **Supported Time Periods**:
   - Seconds: "Xs" (e.g., "30s")
   - Minutes: "Xm" (e.g., "10m")
   - Hours: "Xh" (e.g., "24h")
   - Days: "Xd" (e.g., "1d", "30d")
   - Months: "Xmo" (e.g., "1mo", "2mo")

4. **Requirements**:
   - Redis required for tracking spend across instances
   - Provider names must be litellm provider names. See [Supported Providers](https://docs.litellm.ai/docs/providers)

## Monitoring Provider Remaining Budget

LiteLLM will emit the following metric on Prometheus to track the remaining budget for each provider

This metric indicates the remaining budget for a provider in dollars (USD)

```
litellm_provider_remaining_budget_metric{api_provider="openai"} 10
```

## Multi-instance setup

If you are using a multi-instance setup, you will need to set the Redis host, port, and password in the `proxy_config.yaml` file. Redis is used to sync the spend across LiteLLM instances.

```yaml
model_list:
    - model_name: gpt-3.5-turbo
      litellm_params:
        model: openai/gpt-3.5-turbo
        api_key: os.environ/OPENAI_API_KEY

router_settings:
  provider_budget_config: 
    openai: 
      budget_limit: 0.000000000001 # float of $ value budget for time period
      time_period: 1d # can be 1d, 2d, 30d, 1mo, 2mo
  
  # 👇 Add this: Set Redis Host, Port, and Password if using multiple instance of LiteLLM
  redis_host: os.environ/REDIS_HOST
  redis_port: os.environ/REDIS_PORT
  redis_password: os.environ/REDIS_PASSWORD

general_settings:
  master_key: sk-1234
```

## Spec for provider_budget_config

The `provider_budget_config` is a dictionary where:
- **Key**: Provider name (string) - Must be a valid [LiteLLM provider name](https://docs.litellm.ai/docs/providers)
- **Value**: Budget configuration object with the following parameters:
  - `budget_limit`: Float value representing the budget in USD
  - `time_period`: Duration string in one of the following formats:
    - Seconds: `"Xs"` (e.g., "30s")
    - Minutes: `"Xm"` (e.g., "10m")
    - Hours: `"Xh"` (e.g., "24h")
    - Days: `"Xd"` (e.g., "1d", "30d")
    - Months: `"Xmo"` (e.g., "1mo", "2mo")

Example structure:
```yaml
provider_budget_config:
  openai:
    budget_limit: 100.0    # $100 USD
    time_period: "1d"      # 1 day period
  azure:
    budget_limit: 500.0    # $500 USD
    time_period: "30d"     # 30 day period
  anthropic:
    budget_limit: 200.0    # $200 USD
    time_period: "1mo"     # 1 month period
  gemini:
    budget_limit: 50.0     # $50 USD
    time_period: "24h"     # 24 hour period
```