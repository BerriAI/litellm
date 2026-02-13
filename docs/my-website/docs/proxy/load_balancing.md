import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Proxy - Load Balancing
Load balance multiple instances of the same model

The proxy will handle routing requests (using LiteLLM's Router). **Set `rpm` in the config if you want maximize throughput**


:::info

For more details on routing strategies / params, see [Routing](../routing.md)

:::

## How Load Balancing Works

LiteLLM automatically distributes requests across multiple deployments of the same model using its built-in router. the proxy routes traffic to optimize performance and reliability.

"simple-shuffle" routing strategy is used by default

### Routing Strategies

| Strategy | Description | When to Use |
|----------|-------------|-------------|
| **simple-shuffle** (recommended) | Randomly distributes requests | General purpose, good for even load distribution |
| **least-busy** | Routes to deployment with fewest active requests | High concurrency scenarios |
| **usage-based-routing** (bad for perf) | Routes to deployment with lowest current usage (RPM/TPM) | When you want to respect rate limits evenly |
| **latency-based-routing** | Routes to fastest responding deployment | Latency-critical applications |
| **cost-based-routing** | Routes to deployment with lowest cost | Cost-sensitive applications |

:::tip Deployment Priority
Use the `order` parameter to prioritize specific deployments. [See Deployment Ordering](#deployment-ordering-priority) for details.
:::


## Quick Start - Load Balancing
#### Step 1 - Set deployments on config

**Example config below**. Here requests with `model=gpt-3.5-turbo` will be routed across multiple instances of `azure/gpt-3.5-turbo`
```yaml
model_list:
  - model_name: gpt-3.5-turbo
    litellm_params:
      model: azure/<your-deployment-name>
      api_base: <your-azure-endpoint>
      api_key: <your-azure-api-key>
      rpm: 6      # Rate limit for this deployment: in requests per minute (rpm)
  - model_name: gpt-3.5-turbo
    litellm_params:
      model: azure/gpt-turbo-small-ca
      api_base: https://my-endpoint-canada-berri992.openai.azure.com/
      api_key: <your-azure-api-key>
      rpm: 6
  - model_name: gpt-3.5-turbo
    litellm_params:
      model: azure/gpt-turbo-large
      api_base: https://openai-france-1234.openai.azure.com/
      api_key: <your-azure-api-key>
      rpm: 1440

router_settings:
  routing_strategy: simple-shuffle # Literal["simple-shuffle", "least-busy", "usage-based-routing","latency-based-routing"], default="simple-shuffle"
  model_group_alias: {"gpt-4": "gpt-3.5-turbo"} # all requests with `gpt-4` will be routed to models with `gpt-3.5-turbo`
  num_retries: 2
  timeout: 30                                  # 30 seconds
  redis_host: <your redis host>                # set this when using multiple litellm proxy deployments, load balancing state stored in redis
  redis_password: <your redis password>
  redis_port: 1992
```

## Enforce Model Rate Limits

Strictly enforce RPM/TPM limits set on deployments. When limits are exceeded, requests are blocked **before** reaching the LLM provider with a `429 Too Many Requests` error.

:::info
By default, `rpm` and `tpm` values are only used for **routing decisions** (picking deployments with capacity). With `enforce_model_rate_limits`, they become **hard limits**.
:::

### Quick Start

```yaml
model_list:
  - model_name: gpt-4
    litellm_params:
      model: openai/gpt-4
      api_key: os.environ/OPENAI_API_KEY
    rpm: 60     # 60 requests per minute
    tpm: 90000  # 90k tokens per minute

router_settings:
  optional_pre_call_checks:
    - enforce_model_rate_limits  # 👈 Enables strict enforcement
```

### How It Works

| Limit Type | Enforcement | Accuracy |
|------------|-------------|----------|
| **RPM** | Hard limit - blocked at exact threshold | 100% accurate |
| **TPM** | Best-effort - may slightly exceed | Blocked when already over limit |

**Why TPM is best-effort:** Token count is unknown until the LLM responds. TPM is checked before each request (blocks if already over), and tracked after (adds actual tokens used).

### Error Response

```json
{
  "error": {
    "message": "Model rate limit exceeded. RPM limit=60, current usage=60",
    "type": "rate_limit_error",
    "code": 429
  }
}
```

Response includes `retry-after: 60` header.

### Multi-Instance Deployment

For multiple LiteLLM proxy instances, add Redis to share rate limit state:

```yaml
router_settings:
  optional_pre_call_checks:
    - enforce_model_rate_limits
  redis_host: redis.example.com
  redis_port: 6379
  redis_password: your-password
```


:::info
Detailed information about [routing strategies can be found here](../routing)
:::

#### Step 2: Start Proxy with config

```shell
$ litellm --config /path/to/config.yaml
```

### Test - Simple Call

Here requests with model=gpt-3.5-turbo will be routed across multiple instances of azure/gpt-3.5-turbo

👉 Key Change: `model="gpt-3.5-turbo"`

**Check the `model_id` in Response Headers to make sure the requests are being load balanced**

<Tabs>

<TabItem value="openai" label="OpenAI Python v1.0.0+">

```python
import openai
client = openai.OpenAI(
    api_key="anything",
    base_url="http://0.0.0.0:4000"
)

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages = [
        {
            "role": "user",
            "content": "this is a test request, write a short poem"
        }
    ]
)

print(response)
```
</TabItem>

<TabItem value="Curl" label="Curl Request">

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
</TabItem>

</Tabs>
### Test - Loadbalancing

In this request, the following will occur:
1. A rate limit exception will be raised 
2. LiteLLM proxy will retry the request on the model group (default retries are 3).

```bash
curl -X POST 'http://0.0.0.0:4000/chat/completions' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer sk-1234' \
-d '{
  "model": "gpt-3.5-turbo",
  "messages": [
        {"role": "user", "content": "Hi there!"}
    ],
    "mock_testing_rate_limit_error": true
}'
```

[**See Code**](https://github.com/BerriAI/litellm/blob/6b8806b45f970cb2446654d2c379f8dcaa93ce3c/litellm/router.py#L2535)


## Load Balancing using multiple litellm instances (Kubernetes, Auto Scaling)

LiteLLM Proxy supports sharing rpm/tpm shared across multiple litellm instances, pass `redis_host`, `redis_password` and `redis_port` to enable this. (LiteLLM will use Redis to track rpm/tpm usage )

Example config

```yaml
model_list:
  - model_name: gpt-3.5-turbo
    litellm_params:
      model: azure/<your-deployment-name>
      api_base: <your-azure-endpoint>
      api_key: <your-azure-api-key>
      rpm: 6      # Rate limit for this deployment: in requests per minute (rpm)
  - model_name: gpt-3.5-turbo
    litellm_params:
      model: azure/gpt-turbo-small-ca
      api_base: https://my-endpoint-canada-berri992.openai.azure.com/
      api_key: <your-azure-api-key>
      rpm: 6
router_settings:
  redis_host: <your redis host>
  redis_password: <your redis password>
  redis_port: 1992
  cache_params:
    type: redis
    max_connections: 100  # maximum Redis connections in the pool; tune based on expected concurrency/load
```

## Router settings on config - routing_strategy, model_group_alias

Expose an 'alias' for a 'model_name' on the proxy server. 

```
model_group_alias: {
  "gpt-4": "gpt-3.5-turbo"
}
```

These aliases are shown on `/v1/models`, `/v1/model/info`, and `/v1/model_group/info` by default.

litellm.Router() settings can be set under `router_settings`. You can set `model_group_alias`, `routing_strategy`, `num_retries`,`timeout` . See all Router supported params [here](https://github.com/BerriAI/litellm/blob/1b942568897a48f014fa44618ec3ce54d7570a46/litellm/router.py#L64)



### Usage

Example config with `router_settings`

```yaml
model_list:
  - model_name: gpt-3.5-turbo
    litellm_params:
      model: azure/<your-deployment-name>
      api_base: <your-azure-endpoint>
      api_key: <your-azure-api-key>

router_settings:
  model_group_alias: {"gpt-4": "gpt-3.5-turbo"} # all requests with `gpt-4` will be routed to models 
```

### Hide Alias Models 

Use this if you want to set-up aliases for:

1. typos
2. minor model version changes
3. case sensitive changes between updates

```yaml
model_list:
  - model_name: gpt-3.5-turbo
    litellm_params:
      model: azure/<your-deployment-name>
      api_base: <your-azure-endpoint>
      api_key: <your-azure-api-key>

router_settings:
  model_group_alias:
    "GPT-3.5-turbo": # alias
      model: "gpt-3.5-turbo"  # Actual model name in 'model_list'
      hidden: true             # Exclude from `/v1/models`, `/v1/model/info`, `/v1/model_group/info`
```

### Complete Spec 

```python
model_group_alias: Optional[Dict[str, Union[str, RouterModelGroupAliasItem]]] = {}


class RouterModelGroupAliasItem(TypedDict):
    model: str
    hidden: bool  # if 'True', don't return on `/v1/models`, `/v1/model/info`, `/v1/model_group/info`
```

## Deployment Ordering (Priority)

Set `order` in `litellm_params` to prioritize deployments. Lower values = higher priority. When multiple deployments share the same `order`, the routing strategy picks among them.

```yaml
model_list:
  - model_name: gpt-4
    litellm_params:
      model: azure/gpt-4-primary
      api_key: os.environ/AZURE_API_KEY
      order: 1  # 👈 Highest priority - always tried first

  - model_name: gpt-4
    litellm_params:
      model: azure/gpt-4-fallback
      api_key: os.environ/AZURE_API_KEY_2
      order: 2  # 👈 Used when order=1 is unavailable

router_settings:
  enable_pre_call_checks: true  # 👈 Required for 'order' to work
```

:::important
The `order` parameter requires `enable_pre_call_checks: true` in `router_settings`.
:::

If `order=1` deployment is unavailable (e.g., rate-limited), the router falls back to `order=2` deployments.

### When You'll See Load Balancing in Action

**Immediate Effects:**

- Different deployments serve subsequent requests (visible in logs)
- Better response times during high traffic

**Observable Benefits:**
- **Higher throughput**: More requests handled simultaneously across deployments
- **Improved reliability**: If one deployment fails, traffic automatically routes to healthy ones
- **Better resource utilization**: Load spread evenly across all available deployments
