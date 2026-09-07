## include_cost_in_streaming_usage

Enable this setting to include the final request cost in streaming responses.

Summary
- Purpose: when `litellm_settings.include_cost_in_streaming_usage` is `true`, the proxy will inject the final usage/cost object into the last streaming chunk when the client requests `stream_options.include_usage: true`.

How to enable

In your proxy config (example):

```yaml
litellm_settings:
  include_cost_in_streaming_usage: true
```

How to request usage in streaming calls

When making a streaming request to a provider-compatible endpoint, set:

```json
{"stream_options": {"include_usage": true}}
```

Behavior notes
- This setting only affects streaming responses when both the global flag (`litellm_settings.include_cost_in_streaming_usage`) and the per-request `stream_options.include_usage` are enabled.
- For non-streaming requests the `x-litellm-response-cost` header remains available as before.
- The injected usage object contains the same `usage.cost` breakdown used for non-streaming responses (input/output/cache/discounts) and will appear in the final stream chunk's `usage` field.

Example final stream chunk (simplified):

```json
{
  "id": "chunk-final",
  "object": "chat.completion.chunk",
  "choices": [],
  "usage": {
    "prompt_tokens": 12,
    "completion_tokens": 6,
    "cost": 0.000095,
    "cost_breakdown": {
      "input": 0.000060,
      "output": 0.000035,
      "discount": 0.000000
    }
  }
}
```

Quick curl example (streaming, requesting usage)

```bash
curl -N -X POST "http://localhost:4000/v1/chat/completions" \
  -H "Authorization: Bearer sk-xxx" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role":"user","content":"Say hi"}],
    "stream": true,
    "stream_options": {"include_usage": true}
  }'
```

Notes for reviewers and tests
- See streaming tests that exercise this flow for examples and expected shapes: [tests/local_testing/test_streaming.py](tests/local_testing/test_streaming.py) and [tests/test_litellm/proxy/response_api_endpoints/test_endpoints.py](tests/test_litellm/proxy/response_api_endpoints/test_endpoints.py).
- The implementation reads the same `usage` shape used for non-streaming responses and injects it into the last chunk; reviewers should ensure the example aligns with the live site formatting.

Where to contribute
- The public docs site (`litellm-docs`) hosts user-facing documentation. Add this snippet to the cost-tracking or streaming usage page in `litellm-docs` and link back to the `x-litellm-response-cost` docs if relevant.

Notes for PR authors
- Keep the entry short and add a link to the streaming usage examples/tests if needed. No code changes required in the core repo.
