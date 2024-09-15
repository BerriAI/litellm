# Multiple Instances
Load balance multiple instances of the same model

The proxy will handle routing requests (using LiteLLM's Router). **Set `rpm` in the config if you want maximize throughput**


:::info

For more details on routing strategies / params, see [Routing](../routing.md)

:::

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