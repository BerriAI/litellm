import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Foundry Local

https://github.com/microsoft/Foundry-Local

:::tip

**We support ALL Foundry Local models, just set `model=foundry_local/<any-model-alias>` as a prefix when sending litellm requests**

:::


| Property | Details |
|-------|-------|
| Description | Run AI models on-device with Microsoft Foundry Local. |
| Provider Route on LiteLLM | `foundry_local/` |
| Provider Doc | [Foundry Local ↗](https://learn.microsoft.com/en-us/azure/foundry-local/) |
| Supported OpenAI Endpoints | `/chat/completions` |

## Quick Start

Foundry Local itself is SDK-first and does **not** need to run as a web server for in-process apps. LiteLLM uses the optional OpenAI-compatible web service, so point `FOUNDRY_LOCAL_API_BASE` at `manager.urls[0]/v1`.

<Tabs>
<TabItem value="python" label="Python SDK">

```bash
# Windows (recommended for hardware acceleration)
pip install foundry-local-sdk-winml

# macOS/Linux
pip install foundry-local-sdk
```

```python
import os
from foundry_local_sdk import Configuration, FoundryLocalManager

FoundryLocalManager.initialize(Configuration(app_name="litellm-foundry-local"))
manager = FoundryLocalManager.instance

model = manager.catalog.get_model("qwen2.5-0.5b")
model.download()
model.load()
manager.start_web_service()

os.environ["FOUNDRY_LOCAL_API_BASE"] = f"{manager.urls[0].rstrip('/')}/v1"
print(os.environ["FOUNDRY_LOCAL_API_BASE"])
```

</TabItem>
<TabItem value="javascript" label="JavaScript SDK">

```bash
# Windows (recommended for hardware acceleration)
npm install foundry-local-sdk-winml

# macOS/Linux
npm install foundry-local-sdk
```

```javascript
import { FoundryLocalManager } from "foundry-local-sdk";

const manager = FoundryLocalManager.create({
  appName: "litellm-foundry-local",
});

const model = await manager.catalog.getModel("qwen2.5-0.5b");
await model.download();
await model.load();
manager.startWebService();

const apiBase = `${manager.urls[0].replace(/\/$/, "")}/v1`;
console.log(apiBase);
```

</TabItem>
</Tabs>

:::tip

Foundry Local 1.2 also adds cancellable model and EP downloads (`threading.Event` in Python, `AbortController` in JavaScript) and serves the OpenAI Responses API from the same `/v1` endpoint. LiteLLM currently uses `/chat/completions`, so the local server is optional in general but required for this HTTP integration path.

:::

## API Key
```python
# env variable
os.environ['FOUNDRY_LOCAL_API_BASE']  # e.g. http://127.0.0.1:5272/v1
os.environ['FOUNDRY_LOCAL_API_KEY']   # optional, not required for local use
```

## Sample Usage
```python
from litellm import completion
import os

os.environ['FOUNDRY_LOCAL_API_BASE'] = "http://127.0.0.1:5272/v1"

response = completion(
    model="foundry_local/qwen2.5-0.5b",
    messages=[
        {
            "role": "user",
            "content": "What's the weather like in Boston today in Fahrenheit?",
        }
    ]
)
print(response)
```

## Sample Usage - Streaming
```python
from litellm import completion
import os

os.environ['FOUNDRY_LOCAL_API_BASE'] = "http://127.0.0.1:5272/v1"

response = completion(
    model="foundry_local/qwen2.5-0.5b",
    messages=[
        {
            "role": "user",
            "content": "What's the weather like in Boston today in Fahrenheit?",
        }
    ],
    stream=True,
)

for chunk in response:
    print(chunk)
```

## Usage with DSPy

Foundry Local works seamlessly with [DSPy](https://dspy.ai/) via LiteLLM:

```python
import dspy
import os

os.environ['FOUNDRY_LOCAL_API_BASE'] = "http://127.0.0.1:5272/v1"

lm = dspy.LM("foundry_local/qwen2.5-0.5b")
dspy.configure(lm=lm)
```

## Usage with LiteLLM Proxy Server

Here's how to call a Foundry Local model with the LiteLLM Proxy Server

1. Modify the config.yaml

  ```yaml
  model_list:
    - model_name: my-model
      litellm_params:
        model: foundry_local/qwen2.5-0.5b
        api_base: http://127.0.0.1:5272/v1
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


## Supported Parameters

See [Supported Parameters](../completion/input.md#translated-openai-params) for supported parameters.
