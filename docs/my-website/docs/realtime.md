import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# /realtime

Use this to loadbalance across Azure + OpenAI + xAI and more. 

Supported Providers:
- OpenAI
- Azure
- xAI ([see full docs](/docs/providers/xai_realtime))
- Google AI Studio (Gemini)
- Vertex AI
- Bedrock

## Proxy Usage

### Add model to config 


<Tabs>
<TabItem value="openai" label="OpenAI">

```yaml
model_list:
  - model_name: openai-gpt-4o-realtime-audio
    litellm_params:
      model: openai/gpt-4o-realtime-preview-2024-10-01
      api_key: os.environ/OPENAI_API_KEY
    model_info:
      mode: realtime
```
</TabItem>
<TabItem value="openai+azure" label="OpenAI + Azure">

```yaml
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: azure/gpt-4o-realtime-preview
      api_key: os.environ/AZURE_SWEDEN_API_KEY
      api_base: os.environ/AZURE_SWEDEN_API_BASE

  - model_name: openai-gpt-4o-realtime-audio
    litellm_params:
      model: openai/gpt-4o-realtime-preview-2024-10-01
      api_key: os.environ/OPENAI_API_KEY
```

</TabItem>
<TabItem value="xai" label="xAI Grok Voice Agent">

```yaml
model_list:
  - model_name: grok-voice-agent
    litellm_params:
      model: xai/grok-4-1-fast-non-reasoning
      api_key: os.environ/XAI_API_KEY
    model_info:
      mode: realtime
```

**[See full xAI Realtime documentation →](/docs/providers/xai_realtime)**

</TabItem>
</Tabs>

### Start proxy 

```bash
litellm --config /path/to/config.yaml 

# RUNNING on http://0.0.0.0:8000
```

### Test 

Run this script using node - `node test.js`

```js
// test.js
const WebSocket = require("ws");

const url = "ws://0.0.0.0:4000/v1/realtime?model=openai-gpt-4o-realtime-audio";
// const url = "wss://my-endpoint-sweden-berri992.openai.azure.com/openai/realtime?api-version=2024-10-01-preview&deployment=gpt-4o-realtime-preview";
const ws = new WebSocket(url, {
    headers: {
        "api-key": `sk-1234`,
        "OpenAI-Beta": "realtime=v1",
    },
});

ws.on("open", function open() {
    console.log("Connected to server.");
    ws.send(JSON.stringify({
        type: "response.create",
        response: {
            modalities: ["text"],
            instructions: "Please assist the user.",
        }
    }));
});

ws.on("message", function incoming(message) {
    console.log(JSON.parse(message.toString()));
});

ws.on("error", function handleError(error) {
    console.error("Error: ", error);
});
```

## Guardrails

You can apply [LiteLLM guardrails](https://docs.litellm.ai/docs/proxy/guardrails/quick_start) to realtime sessions.

### Set guardrails on a key or team

The easiest production setup — attach guardrails to a virtual key or team so they always apply automatically, without any client-side changes.

See [Virtual Keys → Guardrails](https://docs.litellm.ai/docs/proxy/virtual_keys#guardrails) and [Teams → Guardrails](https://docs.litellm.ai/docs/proxy/team_budgets).

### Pass guardrails dynamically (easy testing)

Pass `guardrails` as a query param when opening the WebSocket.
Useful for testing guardrails without modifying key/team config.

```js
// node test.js
const WebSocket = require("ws");

const guardrails = ["your-guardrail-name"]; // comma-separated list
const url = `ws://0.0.0.0:4000/v1/realtime?model=openai-gpt-4o-realtime-audio&guardrails=${guardrails.join(",")}`;

const ws = new WebSocket(url, {
    headers: {
        "Authorization": "Bearer sk-1234",
    },
});

ws.on("open", function open() {
    console.log("Connected — guardrails active:", guardrails);
});

ws.on("message", function incoming(message) {
    const data = JSON.parse(message);
    if (data.type === "error") {
        // Guardrail block is sent as an error event before the connection closes
        console.error("Guardrail error:", data.error.message);
    }
});

ws.on("close", function close(code, reason) {
    console.log("Closed:", code, reason.toString());
    // code 1011 = blocked by guardrail at pre_call
});
```

Or with Python:

```python
import asyncio
import websockets

async def main():
    url = "ws://0.0.0.0:4000/v1/realtime?model=openai-gpt-4o-realtime-audio&guardrails=your-guardrail-name"
    async with websockets.connect(
        url,
        additional_headers={"Authorization": "Bearer sk-1234"},
    ) as ws:
        print("Connected — guardrail active")
        async for msg in ws:
            import json
            data = json.loads(msg)
            if data["type"] == "error":
                print("Guardrail blocked:", data["error"]["message"])
                break

asyncio.run(main())
```

When a guardrail blocks the request, the proxy sends an `error` event over the WebSocket and then closes the connection:

```json
{
    "type": "error",
    "error": {
        "type": "guardrail_error",
        "message": "Guardrail blocked this request: <reason>"
    }
}
```

## Logging

To prevent requests from being dropped, by default LiteLLM just logs these event types:

- `session.created`
- `response.create`
- `response.done`

You can override this by setting the `logged_real_time_event_types` parameter in the config. For example:

```yaml
litellm_settings:
  logged_real_time_event_types: "*" # Log all events
  ## OR ## 
  logged_real_time_event_types: ["session.created", "response.create", "response.done"] # Log only these event types
```
