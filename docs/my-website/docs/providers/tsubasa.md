# Tsubasa

Tsubasa uses LiteLLM's existing OpenAI-compatible Chat Completions transport

Tsubasa is currently in website-only mode. API access is unavailable, and live
tool-calling and structured-output qualification is pending. These examples are
for use after the service is enabled

## API key

Set `TSUBASA_API_KEY` in the process environment. LiteLLM resolves the public
endpoint automatically; no `api_base` override is required

## Completion

Select a public model with the `tsubasa/` provider prefix

```python
from litellm import completion

response = completion(
    model="tsubasa/tsubasa-pro",
    messages=[{"role": "user", "content": "Explain binary search."}],
    max_tokens=256,
)
```

Use `tsubasa/tsubasa-fast` for the other public model. Both use
`https://api.tsubasa.sh/v1/chat/completions`

## Streaming

Set `stream=True` to receive Chat Completions chunks

```python
stream = completion(
    model="tsubasa/tsubasa-fast",
    messages=[{"role": "user", "content": "Explain a stack."}],
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="")
```

The integration does not declare Responses, embeddings, image, or audio support.
See [Tsubasa's API documentation](https://tsubasa.sh/docs) for availability
