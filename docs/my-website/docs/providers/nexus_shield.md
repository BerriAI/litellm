# Nexus Shield (Sub-10ms Guardrail Proxy)

Nexus Shield is an in-RAM PII sanitization and security guardrail proxy designed for OpenAI/Anthropic APIs with sub-10ms overhead.

## Usage with LiteLLM

You can pass Nexus Shield as the custom `api_base` endpoint in your LiteLLM completion calls.

```python
import litellm

response = litellm.completion(
    model="openai/gpt-4o",
    messages=[{"role": "user", "content": "Hello, my phone is 555-0199"}],
    api_base="https://api.nexusshield.ai/v1",
    api_key="nx_live_YOUR_KEY",
)

print(response)
```

## Features

- **Sub-10ms Latency:** Pattern matching evaluated directly in RAM before reaching LLMs.
- **Zero-Log Infrastructure:** Redacts PII without writing raw payloads to disk.
