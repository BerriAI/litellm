import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Drop Unsupported Params 

Drop unsupported OpenAI params by your LLM Provider.

## Quick Start 

```python 
import litellm 
import os 

# set keys 
os.environ["COHERE_API_KEY"] = "co-.."

litellm.drop_params = True # 👈 KEY CHANGE

response = litellm.completion(
                model="command-r",
                messages=[{"role": "user", "content": "Hey, how's it going?"}],
                response_format={"key": "value"},
            )
```


LiteLLM maps all supported openai params by provider + model (e.g. function calling is supported by anthropic on bedrock but not titan). 

See `litellm.get_supported_openai_params("command-r")` [**Code**](https://github.com/BerriAI/litellm/blob/main/litellm/utils.py#L3584)

If a provider/model doesn't support a particular param, you can drop it. 

## OpenAI Proxy Usage

```yaml
litellm_settings:
    drop_params: true
```

## Pass drop_params in `completion(..)`

Just drop_params when calling specific models 

<Tabs>
<TabItem value="sdk" label="SDK">

```python 
import litellm 
import os 

# set keys 
os.environ["COHERE_API_KEY"] = "co-.."

response = litellm.completion(
                model="command-r",
                messages=[{"role": "user", "content": "Hey, how's it going?"}],
                response_format={"key": "value"},
                drop_params=True
            )
```
</TabItem>
<TabItem value="proxy" label="PROXY">

```yaml
- litellm_params:
    api_base: my-base
    model: openai/my-model
    drop_params: true # 👈 KEY CHANGE
  model_name: my-model
```
</TabItem>
</Tabs>

## Specify params to drop 

To drop specific params when calling a provider (E.g. 'logit_bias' for vllm)

Use `additional_drop_params`

<Tabs>
<TabItem value="sdk" label="SDK">

```python
import litellm 
import os 

# set keys 
os.environ["COHERE_API_KEY"] = "co-.."

response = litellm.completion(
                model="command-r",
                messages=[{"role": "user", "content": "Hey, how's it going?"}],
                response_format={"key": "value"},
                additional_drop_params=["response_format"]
            )
```
</TabItem>
<TabItem value="proxy" label="PROXY">

```yaml
- litellm_params:
    api_base: my-base
    model: openai/my-model
    additional_drop_params: ["response_format"] # 👈 KEY CHANGE
  model_name: my-model
```
</TabItem>
</Tabs>

**additional_drop_params**: List or null - Is a list of openai params you want to drop when making a call to the model.

## Specify allowed openai params in a request

Tell litellm to allow specific openai params in a request. Use this if you get a `litellm.UnsupportedParamsError` and want to allow a param. LiteLLM will pass the param as is to the model.



<Tabs>
<TabItem value="sdk" label="LiteLLM Python SDK">

In this example we pass `allowed_openai_params=["tools"]` to allow the `tools` param.

```python showLineNumbers title="Pass allowed_openai_params to LiteLLM Python SDK"
await litellm.acompletion(
    model="azure/o_series/<my-deployment-name>",
    api_key="xxxxx",
    api_base=api_base,
    messages=[{"role": "user", "content": "Hello! return a json object"}],
    tools=[{"type": "function", "function": {"name": "get_current_time", "description": "Get the current time in a given location.", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "The city name, e.g. San Francisco"}}, "required": ["location"]}}}]
    allowed_openai_params=["tools"],
)
```
</TabItem>
<TabItem value="proxy" label="LiteLLM Proxy">

When using litellm proxy you can pass `allowed_openai_params` in two ways:

1. Dynamically pass `allowed_openai_params` in a request
2. Set `allowed_openai_params` on the config.yaml file for a specific model

#### Dynamically pass allowed_openai_params in a request
In this example we pass `allowed_openai_params=["tools"]` to allow the `tools` param for a request sent to the model set on the proxy.

```python showLineNumbers title="Dynamically pass allowed_openai_params in a request"
import openai
from openai import AsyncAzureOpenAI

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
    ],
    extra_body={ 
        "allowed_openai_params": ["tools"]
    }
)
```

#### Set allowed_openai_params on config.yaml

You can also set `allowed_openai_params` on the config.yaml file for a specific model. This means that all requests to this deployment are allowed to pass in the `tools` param.

```yaml showLineNumbers title="Set allowed_openai_params on config.yaml"
model_list:
  - model_name: azure-o1-preview
    litellm_params:
      model: azure/o_series/<my-deployment-name>
      api_key: xxxxx
      api_base: https://openai-prod-test.openai.azure.com/openai/deployments/o1/chat/completions?api-version=2025-01-01-preview
      allowed_openai_params: ["tools"]
```
</TabItem>
</Tabs>