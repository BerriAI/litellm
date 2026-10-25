# Tokenization

LiteLLM tokenization is available through the SDK and Router. It supports provider-backed tokenization.

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

Omitting `include` returns every representation available from the provider. An explicit empty array returns the count without token IDs or token strings. A nonempty `include` selects a subset and accepts unique entries from:

- `token_ids`, return token IDs.
- `tokens`, return token strings when the provider supports them.

Request both representations with `include=["token_ids", "tokens"]`. A requested representation must be supported. Provider-unavailable representations remain `null`, and unrequested representations are `null`. The response model is the resolved provider model, not a Router alias. The count uses the ordinary encoding path. This API does not change legacy token-count behavior.

`add_special_tokens` is optional and must be a boolean when supplied. Omitting it lets the provider decide.

## Other token APIs

Tokenization is separate from the legacy `token_counter`, `encode`, and `decode` APIs. Those APIs retain their existing behavior and defaults.
