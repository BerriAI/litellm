import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Cloudrift
https://cloudrift.ai/

:::tip

**We support ALL Cloudrift models, just set `model=cloudrift/<any-model-on-cloudrift>` as a prefix when sending LiteLLM requests**

:::

## Table of Contents

- [API Key](#api-key)
- [Chat Models](#chat-models)

## API Key
```python
# env variable
os.environ['CLOUDRIFT_API_KEY']
```

## Sample Usage
```python
from litellm import completion
import os

os.environ['CLOUDRIFT_API_KEY'] = ""
response = completion(
    model="cloudrift/deepseek-ai/DeepSeek-V3.1", 
    messages=[{"role": "user", "content": "write code for saying hi from LiteLLM"}]
)
print(response)
```

## Sample Usage - Streaming
```python
from litellm import completion
import os

os.environ['CLOUDRIFT_API_KEY'] = ""
response = completion(
    model="cloudrift/deepseek-ai/DeepSeek-V3.1", 
    messages=[{"role": "user", "content": "write code for saying hi from LiteLLM"}],
    stream=True
)

for chunk in response:
    print(chunk)
```

## Chat Models
| Model Name       | Function Call                        |
|------------------|--------------------------------------|
| deepseek-ai/DeepSeek-V3.1 | `completion(model="cloudrift/deepseek-ai/DeepSeek-V3.1", messages)` | 
| moonshotai/Kimi-K2-Instruct | `completion(model="cloudrift/moonshotai/Kimi-K2-Instruct", messages)` | 
| deepseek-ai/DeepSeek-R1-0528 | `completion(model="cloudrift/deepseek-ai/DeepSeek-R1-0528", messages)` | 
| deepseek-ai/DeepSeek-V3 | `completion(model="cloudrift/deepseek-ai/DeepSeek-V3", messages)` | 
| Qwen/Qwen3-Next-80B-A3B-Thinking | `completion(model="cloudrift/Qwen/Qwen3-Next-80B-A3B-Thinking", messages)` | 

---

</TabItem>
<TabItem value="proxy" label="PROXY">

1. Add to config.yaml
```yaml
model_list:
  - model_name: deepseek-ai/DeepSeek-V3.1
    litellm_params:
      model: cloudrift/deepseek-ai/DeepSeek-V3.1
      api_key: os.environ/CLOUDRIFT_API_KEY
```

2. Start proxy 

```bash
litellm --config /path/to/config.yaml

# RUNNING on http://0.0.0.0:4000/
```

3. Test it! 

```bash 
curl -L -X POST 'http://0.0.0.0:4000/rerank' \
-H 'Authorization: Bearer sk-1234' \
-H 'Content-Type: application/json' \
-d '{
    "model": "deepseek-ai/DeepSeek-V3.1",
    "query": "What is the capital of France?"
}'
```

</TabItem>
</Tabs>


### Provider-specific parameters
Pass any Cloudrift-specific parameters as keyword arguments to the `completion` function, e.g.

```python
response = completion(
    model="cloudrift/deepseek-ai/DeepSeek-V3.1",
    messages=[{"role": "user", "content": "write code for saying hi from LiteLLM"}],
    my_custom_param="my_custom_value"  # any Cloudrift-specific parameters
)
```
