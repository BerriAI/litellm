---
id: cursor
title: Cursor Endpoint (/cursor/chat/completions)
description: Accept Responses API input from Cursor and return OpenAI Chat Completions output
---

LiteLLM provides a Cursor-specific endpoint to make Cursor IDE work seamlessly with the LiteLLM Proxy when using BYOK + custom `base_url`.

- Accepts Requests in OpenAI Responses API input format (Cursor sends this)
- Returns Responses in OpenAI Chat Completions format (Cursor expects this)
- Supports streaming and non‑streaming

## Endpoint

- Path: `/cursor/chat/completions`
- Auth: Standard LiteLLM Proxy auth (`Authorization: Bearer <key>`)
- Behavior: Internally routes to LiteLLM `/responses` flow and transforms output to Chat Completions

## Why this exists

When setting up Cursor with BYOK against a custom `base_url`, Cursor sends requests to the Chat Completions endpoint but in the OpenAI Responses API input shape. Without translation, Cursor won’t display streamed output. This endpoint bridges the formats:

- Input: Responses API (`input`, tool calls, etc.)
- Output: Chat Completions (`choices`, `delta`, `finish_reason`, etc.)

## Usage

### Non-streaming

```bash
curl -X POST http://localhost:4000/cursor/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{
    "model": "gpt-4o",
    "input": [{"role": "user", "content": "Hello"}]
  }'
```

Example response (shape):

```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1733333333,
  "model": "gpt-4o",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! How can I help you?"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 8,
    "total_tokens": 18
  }
}
```

### Streaming

```bash
curl -N -X POST http://localhost:4000/cursor/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{
    "model": "gpt-4o",
    "input": [{"role": "user", "content": "Hello"}],
    "stream": true
  }'
```

- Server-Sent Events (SSE)
- Emits `chat.completion.chunk` deltas (`choices[].delta`) and ends with `data: [DONE]`

## Configuration

No special configuration is required beyond your normal LiteLLM Proxy setup. Ensure that:

- Your `config.yaml` includes the models you want to call via this endpoint
- Your Cursor project uses your LiteLLM Proxy `base_url` and a valid API key

## Notes

- Only this page documents the Cursor endpoint. The native `/responses` docs remain unchanged.
- This endpoint is intended specifically for Cursor’s request/response expectations. Other clients should continue to use `/v1/chat/completions` or `/v1/responses` as appropriate.


