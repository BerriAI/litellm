# Vertex Realtime API

## Feature Matrix

| Feature | Description | Comments |
| --- | --- | --- |
| Proxy | ✅ | Works via LiteLLM proxy |
| SDK | ⌛️ | Experimental access via `litellm._arealtime` |

---

## Proxy Usage

### Add model to config

```yaml
model_list:
  - model_name: "vertex_gemini_live_text"        # your friendly name
    litellm_params:
      model: vertex_ai/gemini-2.0-flash-live-preview-04-09
    model_info:
      mode: realtime

### Start proxy 

```bash
litellm --config /path/to/config.yaml 

# RUNNING on http://0.0.0.0:8000
```

### Test 

Run this script using node - `node test.js`

```js
//test.js
const WebSocket = require("ws");
const url = "ws://0.0.0.0:4000/v1/realtime?model=vertex_gemini_live_text";

const ws = new WebSocket(url, {
  headers: {
    "api-key": process.env.LITELLM_API_KEY || "sk-key-goes-here",
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
    ws.send(JSON.stringify({ type: "input_text", text: "hey there!" }));
});

ws.on("message", function incoming(message) {
    console.log(JSON.parse(message.toString()));
});

ws.on("error", function handleError(error) {
    console.error("Error: ", error);
});
```

## Limitations 

- Tool calling: not supported yet.
- Audio transcription: not supported yet.

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

## Supported Session Params

The proxy maps these OpenAI fields to Vertex setup:
- `instructions`
- `temperature`
- `max_output_tokens` / `max_response_output_tokens`
- `modalities` (`TEXT`, `AUDIO`, `IMAGE`, `VIDEO`)
- `turn_detection`