import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Firmware.ai
https://docs.firmware.ai

:::tip

**We support ALL Firmware.ai models, just set `model=firmware/<any-model-on-firmware>` as a prefix when sending litellm requests**

:::

Firmware.ai provides unified access to multiple AI providers (OpenAI, Anthropic, Google, xAI, DeepSeek, Cerebras) through a single API endpoint with subscription-based pricing.

## API Key
```python
# env variable
os.environ['FIRMWARE_API_KEY']
```

## Sample Usage
```python
from litellm import completion
import os

os.environ['FIRMWARE_API_KEY'] = ""
response = completion(
    model="firmware/openai/gpt-4o",
    messages=[
       {"role": "user", "content": "hello from litellm"}
   ],
)
print(response)
```

## Sample Usage - Streaming
```python
from litellm import completion
import os

os.environ['FIRMWARE_API_KEY'] = ""
response = completion(
    model="firmware/anthropic/claude-opus-4-5",
    messages=[
       {"role": "user", "content": "hello from litellm"}
   ],
    stream=True
)

for chunk in response:
    print(chunk)
```


## Supported Models - ALL Firmware.ai Models Supported!
We support ALL Firmware.ai models, just set `firmware/` as a prefix when sending completion requests

### OpenAI Models
| Model Name | Function Call |
|------------|---------------|
| gpt-5.2 | `completion(model="firmware/openai/gpt-5.2", messages)` |
| gpt-5 | `completion(model="firmware/openai/gpt-5", messages)` |
| gpt-5-mini | `completion(model="firmware/openai/gpt-5-mini", messages)` |
| gpt-5-nano | `completion(model="firmware/openai/gpt-5-nano", messages)` |
| gpt-4o | `completion(model="firmware/openai/gpt-4o", messages)` |
| gpt-4o-mini | `completion(model="firmware/openai/gpt-4o-mini", messages)` |

### Anthropic Models
| Model Name | Function Call |
|------------|---------------|
| claude-opus-4-5 | `completion(model="firmware/anthropic/claude-opus-4-5", messages)` |
| claude-sonnet-4-5-20250929 | `completion(model="firmware/anthropic/claude-sonnet-4-5-20250929", messages)` |
| claude-haiku-4-5-20251001 | `completion(model="firmware/anthropic/claude-haiku-4-5-20251001", messages)` |

### Google Models
| Model Name | Function Call |
|------------|---------------|
| gemini-3-pro-preview | `completion(model="firmware/google/gemini-3-pro-preview", messages)` |
| gemini-3-flash-preview | `completion(model="firmware/google/gemini-3-flash-preview", messages)` |
| gemini-2.5-pro | `completion(model="firmware/google/gemini-2.5-pro", messages)` |
| gemini-2.5-flash | `completion(model="firmware/google/gemini-2.5-flash", messages)` |

### xAI Models
| Model Name | Function Call |
|------------|---------------|
| grok-4-fast-reasoning | `completion(model="firmware/xai/grok-4-fast-reasoning", messages)` |
| grok-4-fast-non-reasoning | `completion(model="firmware/xai/grok-4-fast-non-reasoning", messages)` |
| grok-code-fast-1 | `completion(model="firmware/xai/grok-code-fast-1", messages)` |

### DeepSeek Models
| Model Name | Function Call |
|------------|---------------|
| deepseek-chat | `completion(model="firmware/deepseek/deepseek-chat", messages)` |
| deepseek-reasoner | `completion(model="firmware/deepseek/deepseek-reasoner", messages)` |

### Cerebras Models
| Model Name | Function Call |
|------------|---------------|
| gpt-oss-120b | `completion(model="firmware/cerebras/gpt-oss-120b", messages)` |
| zai-glm-4.7 | `completion(model="firmware/cerebras/zai-glm-4.7", messages)` |


## Usage with LiteLLM Proxy Server

Here's how to call a Firmware.ai model with the LiteLLM Proxy Server

1. Modify the config.yaml

  ```yaml
  model_list:
    - model_name: my-model
      litellm_params:
        model: firmware/openai/gpt-4o  # add firmware/ prefix to route as Firmware provider
        api_key: api-key               # api key to send your model
  ```


2. Start the proxy

  ```bash
  $ litellm --config /path/to/config.yaml
  ```

3. Send Request to LiteLLM Proxy Server

  <Tabs>

  <TabItem value="openai" label="OpenAI Python v1.0.0+">

  ```python
  import openai
  client = openai.OpenAI(
      api_key="sk-1234",             # pass litellm proxy key, if you're using virtual keys
      base_url="http://0.0.0.0:4000" # litellm-proxy-base url
  )

  response = client.chat.completions.create(
      model="my-model",
      messages = [
          {
              "role": "user",
              "content": "what llm are you"
          }
      ],
  )

  print(response)
  ```
  </TabItem>

  <TabItem value="curl" label="curl">

  ```shell
  curl --location 'http://0.0.0.0:4000/chat/completions' \
      --header 'Authorization: Bearer sk-1234' \
      --header 'Content-Type: application/json' \
      --data '{
      "model": "my-model",
      "messages": [
          {
          "role": "user",
          "content": "what llm are you"
          }
      ],
  }'
  ```
  </TabItem>

  </Tabs>

