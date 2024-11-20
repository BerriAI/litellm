import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Provider Budget Routing
Use this to set budgets for LLM Providers - example $100/day for OpenAI, $100/day for Azure.

```yaml
model_list:
    - model_name: gpt-3.5-turbo
      litellm_params:
        model: openai/gpt-3.5-turbo
        api_key: os.environ/OPENAI_API_KEY
    - model_name: gpt-3.5-turbo
      litellm_params:
        model: azure/chatgpt-functioncalling
        api_key: os.environ/AZURE_API_KEY
        api_version: os.environ/AZURE_API_VERSION
        api_base: os.environ/AZURE_API_BASE

router_settings:
  redis_host: <your-redis-host>
  redis_password: <your-redis-password>
  redis_port: <your-redis-port>
  provider_budget_config: 
	openai: 
		budget_limit: 0.000000000001 # float of $ value budget for time period
		time_period: 1d # can be 1d, 2d, 30d 
	azure:
		budget_limit: 100
		time_period: 1d
	anthropic:
		budget_limit: 100
		time_period: 10d
	vertexai:
		budget_limit: 100
		time_period: 12d
	gemini:
		budget_limit: 100
		time_period: 12d

general_settings:
  master_key: sk-1234
```


#### How provider-budget-routing works

1. **Budget Tracking**: 
   - Uses Redis to track spend for each provider
   - Tracks spend over specified time periods (e.g., "1d", "30d")
   - Automatically resets spend after time period expires

2. **Routing Logic**:
   - Routes requests to providers under their budget limits
   - Skips providers that have exceeded their budget
   - If all providers exceed budget, raises an error

3. **Supported Time Periods**:
   - Format: "Xd" where X is number of days
   - Examples: "1d" (1 day), "30d" (30 days)

4. **Requirements**:
   - Redis required for tracking spend across instances
   - Provider names must be litellm provider names. See [Supported Providers](https://docs.litellm.ai/docs/providers)
