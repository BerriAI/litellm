# Gemini Realtime API - Google AI Studio

| Feature | Description | Comments |
| --- | --- | --- |
| Proxy | ✅ |  |
| SDK | ⌛️ | Experimental access via `litellm._arealtime`. |


## Proxy Usage

### Add model to config 

```yaml
model_list:
  - model_name: "gemini-2.0-flash"
    litellm_params:
      model: gemini/gemini-2.0-flash-live-001
    model_info:
      mode: realtime
```

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

const url = "ws://0.0.0.0:4000/v1/realtime?model=openai-gemini-2.0-flash";

const ws = new WebSocket(url, {
    headers: {
        "api-key": `${LITELLM_API_KEY}`,
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

## Limitations 

- Does not support audio transcription. 
- Does not support tool calling 

## Supported OpenAI Realtime Events

- `session.created`
- `response.created`
- `response.output_item.added`
- `conversation.item.created`
- `response.content_part.added`
- `response.text.delta`
- `response.audio.delta`
- `response.text.done`
- `response.audio.done`
- `response.content_part.done`
- `response.output_item.done`
- `response.done`



## [Supported Session Params](https://github.com/BerriAI/litellm/blob/e87b536d038f77c2a2206fd7433e275c487179ee/litellm/llms/gemini/realtime/transformation.py#L155)

## More Examples
### [Gemini Realtime API with Audio Input/Output](../../../docs/tutorials/gemini_realtime_with_audio)