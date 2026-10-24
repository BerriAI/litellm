# Tokenization

LiteLLM tokenization is available through the SDK, Router, and proxy HTTP routes. It supports provider-backed tokenization.

## SDK

```python
import os

import litellm

result = litellm.tokenize(
    model="hosted_vllm/your-model",
    prompt="Hello, world",
    api_base=os.environ["HOSTED_VLLM_API_BASE"],
    api_key=os.environ["HOSTED_VLLM_API_KEY"],
    include=["token_ids"],
)
print(result.model, result.count, result.token_ids)
```

Use `await litellm.atokenize(...)` for asynchronous calls. The public types are available from `litellm.types.tokenize`.

`include` defaults to an empty array, which returns the count without token IDs or token strings. It accepts unique entries from:

- `token_ids`, return token IDs.
- `tokens`, return token strings when the provider supports them.

Request both representations with `include=["token_ids", "tokens"]`. A requested representation must be supported. Unrequested representations are `null`. The response model is the resolved provider model, not a Router alias. The count uses the ordinary encoding path. This API does not change legacy token-count behavior.

`add_special_tokens` is optional and must be a boolean when supplied. Omitting it lets the provider decide.

## Proxy

Send a request to either `/tokenize` or `/v1/tokenize`:

```bash
curl http://localhost:4000/tokenize \
  -H 'Authorization: Bearer sk-...' \
  -H 'Content-Type: application/json' \
  -d '{"model":"tokenizer","prompt":"Hello, world","include":["token_ids"]}'
```

The proxy resolves the model deployment and applies its configured provider. Client credentials and provider options remain subject to the proxy's existing authorization policy.

This endpoint does not change `token_counter`, `count_tokens`, or other legacy token-count APIs.

## Other token APIs

Tokenization is separate from the legacy `token_counter`, `encode`, and `decode` APIs. Those APIs retain their existing behavior and defaults.
