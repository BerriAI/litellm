import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Provider specific Wildcard routing 

**Proxy all models from a provider**

Use this if you want to **proxy all models from a specific provider without defining them on the config.yaml**

## Step 1. Define provider specific routing 

<Tabs>
<TabItem value="sdk" label="SDK">

```python
from litellm import Router

router = Router(
    model_list=[
        {
            "model_name": "anthropic/*",
            "litellm_params": {
                "model": "anthropic/*",
                "api_key": os.environ["ANTHROPIC_API_KEY"]
            }
        }, 
        {
            "model_name": "groq/*",
            "litellm_params": {
                "model": "groq/*",
                "api_key": os.environ["GROQ_API_KEY"]
            }
        }, 
        {
            "model_name": "fo::*:static::*", # all requests matching this pattern will be routed to this deployment, example: model="fo::hi::static::hi" will be routed to deployment: "openai/fo::*:static::*"
            "litellm_params": {
                "model": "openai/fo::*:static::*",
                "api_key": os.environ["OPENAI_API_KEY"]
            }
        }
    ]
)
```

</TabItem>
<TabItem value="proxy" label="PROXY">

**Step 1** - define provider specific routing on config.yaml
```yaml
model_list:
  # provider specific wildcard routing
  - model_name: "anthropic/*"
    litellm_params:
      model: "anthropic/*"
      api_key: os.environ/ANTHROPIC_API_KEY
  - model_name: "groq/*"
    litellm_params:
      model: "groq/*"
      api_key: os.environ/GROQ_API_KEY
  - model_name: "fo::*:static::*" # all requests matching this pattern will be routed to this deployment, example: model="fo::hi::static::hi" will be routed to deployment: "openai/fo::*:static::*"
    litellm_params:
      model: "openai/fo::*:static::*"
      api_key: os.environ/OPENAI_API_KEY
```
</TabItem>
</Tabs>

## [PROXY-Only] Step 2 - Run litellm proxy 

```shell
$ litellm --config /path/to/config.yaml
```

## Step 3 - Test it 

<Tabs>  
<TabItem value="sdk" label="SDK">

```python
from litellm import Router

router = Router(model_list=...)

# Test with `anthropic/` - all models with `anthropic/` prefix will get routed to `anthropic/*`
resp = completion(model="anthropic/claude-3-sonnet-20240229", messages=[{"role": "user", "content": "Hello, Claude!"}])
print(resp)

# Test with `groq/` - all models with `groq/` prefix will get routed to `groq/*`
resp = completion(model="groq/llama3-8b-8192", messages=[{"role": "user", "content": "Hello, Groq!"}])
print(resp)

# Test with `fo::*::static::*` - all requests matching this pattern will be routed to `openai/fo::*:static::*`
resp = completion(model="fo::hi::static::hi", messages=[{"role": "user", "content": "Hello, Claude!"}])
print(resp)
```

</TabItem>
<TabItem value="proxy" label="PROXY">

Test with `anthropic/` - all models with `anthropic/` prefix will get routed to `anthropic/*`
```bash
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{
    "model": "anthropic/claude-3-sonnet-20240229",
    "messages": [
      {"role": "user", "content": "Hello, Claude!"}
    ]
  }'
```

Test with `groq/` - all models with `groq/` prefix will get routed to `groq/*`
```shell
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{
    "model": "groq/llama3-8b-8192",
    "messages": [
      {"role": "user", "content": "Hello, Claude!"}
    ]
  }'
```

Test with `fo::*::static::*` - all requests matching this pattern will be routed to `openai/fo::*:static::*`
```shell
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{
    "model": "fo::hi::static::hi",
    "messages": [
      {"role": "user", "content": "Hello, Claude!"}
    ]
  }'
```

</TabItem>
</Tabs>


## [[PROXY-Only] Control Wildcard Model Access](./proxy/model_access#-control-access-on-wildcard-models)