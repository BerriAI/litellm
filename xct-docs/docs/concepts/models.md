---
sidebar_position: 2
---

# Models

A **model** is a stable name (e.g. `deepseek-v3.2`, `gpt-4o`,
`anthropic/claude-3-7-sonnet`) the proxy routes to one or more provider
deployments. Wire shape is OpenAI's `/v1/chat/completions`, plus extensions.

## Discovery

`GET /v1/models` returns the standard OpenAI list shape PLUS:

```json
{
  "id": "gpt-4o",
  "object": "model",
  "owned_by": "openai",
  "capabilities": {
    "vision": true,
    "function_calling": true,
    "structured_output": true,
    "prompt_caching": true,
    "pdf_input": false,
    "web_search": false,
    "audio_input": false,
    "audio_output": false
  },
  "context_window": 128000,
  "max_output_tokens": 4096
}
```

The `capabilities` flags come from the same `litellm.utils.supports_*`
helpers that govern the proxy's own request validation. Use them on the
SDK side to gate features (`if model.capabilities.vision`) instead of
maintaining a parallel matrix.

## Calling

```python
from xct_litellm import XctClient

xct = XctClient(base_url="https://api.xct.test", access_token="sk-...")
reply = xct.chat.completions.create(
    model="deepseek-v3.2",
    messages=[{"role": "user", "content": "hello"}],
)
```

## Streaming

```python
for chunk in xct.chat.completions.create(
    model="deepseek-v3.2",
    messages=[...],
    stream=True,
):
    print(chunk["choices"][0]["delta"].get("content", ""), end="")
```

Server flips to `text/event-stream` when the SDK sends
`Accept: text/event-stream` (which it does for `stream=True`). NDJSON is
the fallback wire format.

## Provider routing

The model name is opaque to the SDK. The proxy maps it to a deployment
(or a list, with weighted fallback per S4's upstream merge). Add models
via the dashboard's **Add Model** page or `POST /model/new` — that's an
admin operation, not part of the consumer surface.

## Scope

`/v1/models` is **caller-scoped** by default — the response only includes
models the caller's key (and team, and app's `capability_scope_id`) can
invoke. Admin keys see the full registry.

## See also

- **[Capability discovery](./overview.md)** — `/v1/capabilities` returns
  the same model list inside a richer envelope.
- **[App-Tenancy](./app-tenancy.md)** — how `capability_scope_id`
  narrows what an app's tokens see.
