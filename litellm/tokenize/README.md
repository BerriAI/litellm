# Tokenization

LiteLLM tokenization is available through the SDK, Router, and proxy HTTP routes. It supports provider-backed tokenization and explicitly configured local tokenizers.

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

Request both representations with `include=["token_ids", "tokens"]`. A requested representation must be supported. Unrequested representations are `null`. The response model is the resolved provider model or the configured local tokenizer ID, not a Router alias. The count uses the ordinary encoding path. This API does not change legacy token-count behavior.

`add_special_tokens` is optional and must be a boolean when supplied. Omitting it lets the provider or local tokenizer configuration decide.

## Local tokenizer

Local tokenization requires an explicit backend configuration. It processes the prompt from a configured artifact and does not make a provider request.

```python
import litellm
from litellm.types.tokenize import validate_tokenize_backend

backend = validate_tokenize_backend(
    {
        "type": "local",
        "config": {
            "id": "word-level",
            "revision": "v1",
            "engine": {"type": "hugging_face"},
            "artifact": {
                "type": "inline",
                "content": tokenizer_json,
            },
        },
    }
)

result = litellm.tokenize(
    model="local-word-level",
    prompt="Hello, world",
    backend=backend,
    include=["token_ids", "tokens"],
)
```

The local configuration contains an `id`, `revision`, and an `engine`. Hugging Face uses `{"type": "hugging_face"}` and a Hugging Face tokenizer JSON artifact. Tiktoken uses `{"type": "tiktoken", "encoding": "cl100k_base"}` with a compatible ranks artifact. An artifact is either inline `{"type": "inline", "content": ...}` or a local file `{"type": "file", "path": ..., "sha256": ...}`. File artifacts are checked against the supplied SHA-256 digest. Optional `special_text` values are `literal` and `recognize`; `add_special_tokens` controls special-token insertion where supported.

The local backend works from the configured artifact and does not download tokenizer data. Local tokenization accepts text prompts only. Asynchronous calls use native worker capacity for artifact loading and encoding so the Python event loop can continue running other tasks.

Local configurations can also be set on a Router deployment. A proxy deployment can select the same backend in its model configuration; clients send the configured model name and prompt, not local artifact paths or backend configuration.

## Proxy

Send a request to either `/tokenize` or `/v1/tokenize`:

```bash
curl http://localhost:4000/tokenize \
  -H 'Authorization: Bearer sk-...' \
  -H 'Content-Type: application/json' \
  -d '{"model":"tokenizer","prompt":"Hello, world","include":["token_ids"]}'
```

The proxy resolves the model deployment and applies its configured provider or local backend. Clients cannot provide local artifact configuration. Client credentials and provider options remain subject to the proxy's existing authorization policy.

This endpoint does not change `token_counter`, `count_tokens`, or other legacy token-count APIs.

## Other token APIs

Tokenization is separate from the legacy `token_counter`, `encode`, and `decode` APIs. Those APIs retain their existing behavior and defaults.
