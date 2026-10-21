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

Use [Foundry Local Python SDK 2.2.0](https://pypi.org/project/foundry-local-sdk/2.2.0/) to run models on-device. Foundry Local supports in-process inference, but LiteLLM uses its optional OpenAI-compatible HTTP service. Start that service and point `FOUNDRY_LOCAL_API_BASE` at its `/v1` URL.

<Tabs>
<TabItem value="python" label="Python SDK">

```bash
pip install foundry-local-sdk==2.2.0
```

```python
import os
from foundry_local_sdk import Configuration, FoundryLocalManager
from litellm import completion

FoundryLocalManager.initialize(Configuration(app_name="litellm-foundry-local"))
manager = FoundryLocalManager.instance

ep_result = manager.download_and_register_eps()
if not ep_result.success:
    raise RuntimeError(f"Execution provider setup failed: {ep_result.failed_eps}")

model = manager.catalog.get_model("qwen2.5-0.5b")
model.download()
model.load()
manager.start_web_service()

os.environ["FOUNDRY_LOCAL_API_BASE"] = f"{manager.urls[0].rstrip('/')}/v1"
try:
    response = completion(
        model=f"foundry_local/{model.id}",
        messages=[{"role": "user", "content": "What is the golden ratio?"}],
    )
    print(response.choices[0].message.content)
finally:
    manager.stop_web_service()
    model.unload()
    manager.close()
```

</TabItem>
<TabItem value="javascript" label="JavaScript SDK">

```bash
npm install foundry-local-sdk
```

```javascript
import { FoundryLocalManager } from "foundry-local-sdk";

const manager = FoundryLocalManager.create({
  appName: "litellm-foundry-local",
});

const epResult = await manager.downloadAndRegisterEps();
if (!epResult.success) {
  throw new Error(`Execution provider setup failed: ${epResult.failedEps}`);
}

const model = await manager.catalog.getModel("qwen2.5-0.5b");
await model.download();
await model.load();
manager.startWebService();

const apiBase = `${manager.urls[0].replace(/\/$/, "")}/v1`;
console.log(apiBase);
console.log(`Use model: foundry_local/${model.id}`);
```

Keep this process running while LiteLLM uses the service. Configure the printed URL in the Python or proxy process, since JavaScript cannot set another process's environment variables. When finished, stop the service with `manager.stopWebService()` and unload the model with `await model.unload()`.

</TabItem>
</Tabs>

### SDK 2.2.0 requirements

The unified `foundry-local-sdk` Python wheel supports Windows x64, Linux x64/ARM64, and macOS Apple silicon. It requires Python 3.11 through 3.14. Windows wheels include the WinML runtime, so the separate `foundry-local-sdk-winml` package is no longer needed.

Call `download_and_register_eps()` before selecting and loading a model to make hardware execution providers available. Catalog access no longer waits for execution-provider downloads.

The service uses a dynamically assigned local port by default. Read `manager.urls` after starting it instead of assuming port 5272. For a separate proxy process that needs a fixed address, initialize Python with `Configuration(app_name="litellm-foundry-local", web=Configuration.WebService(urls="http://127.0.0.1:5272"))` and keep that service process running.

LiteLLM does not import or install the Foundry Local SDK. The SDK's Python requirement applies only to the process hosting Foundry Local, and existing OpenAI-compatible Foundry Local endpoints continue to work. This provider uses chat completions for both streaming and non-streaming requests.

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
      ]
  }'
  ```
  </TabItem>

  </Tabs>


## Supported Parameters

See [Supported Parameters](../completion/input.md#translated-openai-params) for supported parameters.
