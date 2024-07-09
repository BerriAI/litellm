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