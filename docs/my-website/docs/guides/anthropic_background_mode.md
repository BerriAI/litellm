# Stream OpenAI/Anthropic/Gemini/etc. in Ruby  

Imaging the case where services are running frameworks that does not support multithreading (for example, your company runs ruby service on single thread framework), you want to stream responses, but don't want to block I/O for others since it is a single thread. Native SSE streaming won't work in this case. LiteLLM supports background mode where user can use litellm to poll partial responses periodically to simulate streaming.


Requirements:
- Redis cache must be enabled
- /responses API endpoint must be used 

## How it works

Think of this as a producer–consumer pattern:

**Producer**: LiteLLM runs the LLM request in the background and incrementally writes chunks to Redis.

**Consumer**: Your app periodically polls LiteLLM to fetch the latest partial response.

## Usage

1. Setup config.yaml

```yaml
model_list:
  - model_name: claude-sonnet-4-5-20250929
    litellm_params:
      model: anthropic/claude-sonnet-4-5-20250929

litellm_settings:
  cache: true
  cache_params:
    type: redis
    ttl: 3600
    host: "127.0.0.1"
    port: "6379"
  responses:
    background_mode:
      polling_via_cache: "all"
      ttl: 3600
```

2. Start LiteLLM Proxy

```bash
litellm --config /path/to/config.yaml
```

3. Make request

```bash
curl http://0.0.0.0:4000/v1/responses \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{
    "model": "claude-sonnet-4-5-20250929",
    "input": "Tell me a three sentence bedtime story about a unicorn.",
    "background": true
  }'
```

Expected Response:

```bash
{
  "id": "litellm_poll_adff0089-f9d3-4135-b49e-f582597deedd",
  "object": "response",
  "status": "queued",
  "output": [],
  "usage": null,
  "metadata": {},
  "created_at": 1764913614
}
```

Get response curl cmd:

```bash
curl http://0.0.0.0:4000/v1/responses/litellm_poll_adff0089-f9d3-4135-b49e-f582597deedd \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234"
```

Response: 

**Keep polling until the response is completed.**

```bash
{
  "id": "litellm_poll_adff0089-f9d3-4135-b49e-f582597deedd",
  "created_at": 1764913614,
  "error": null,
  "incomplete_details": null,
  "instructions": null,
  "metadata": {},
  "model": "claude-sonnet-4-5-20250929",
  "object": "response",
  "output": [
    {
      "id": "rs_0b467fde531ff8fb00693271cf4ec88196aee3b64030ecc23d",
      "summary": [],
      "type": "reasoning"
    },
    {
      "id": "msg_0b467fde531ff8fb00693271d2885c8196b2ccf625b4560125",
      "content": [
        {
          "annotations": [],
          "text": "Under a moonlit sky, a gentle unicorn named Lumen wandered through a whispering forest, leaving silver hoofprints that glowed softly in the moss. She paused beside a sleepy brook and lowered her horn, and the ripples turned into twinkling stars that drifted up to join the night. With the forest tucked beneath a quilt of starlight, Lumen closed her eyes and dreamed of new paths to light when morning came.",
          "type": "output_text",
          "logprobs": []
        }
      ],
      "role": "assistant",
      "status": "completed",
      "type": "message"
    }
  ],
  "parallel_tool_calls": true,
  "temperature": 1,
  "tool_choice": "auto",
  "tools": [],
  "top_p": 1,
  "max_output_tokens": null,
  "previous_response_id": null,
  "reasoning": {
    "effort": "medium"
  },
  "status": "completed",
  "text": {
    "format": {
      "type": "text"
    },
    "verbosity": "medium"
  },
  "truncation": "disabled",
  "usage": {
    "input_tokens": 17,
    "input_tokens_details": {
      "cached_tokens": 0
    },
    "output_tokens": 287,
    "output_tokens_details": {
      "reasoning_tokens": 192
    },
    "total_tokens": 304
  },
  "user": null,
  "store": true
}
```