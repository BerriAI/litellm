# HPC-AI

[HPC-AI](https://api.hpc-ai.com) provides an OpenAI-compatible inference API at `https://api.hpc-ai.com/inference/v1`.

:::tip

Use the `hpc_ai/` prefix with the upstream model id (for example `hpc_ai/minimax/minimax-m2.5`). LiteLLM strips the prefix and forwards the remainder as the OpenAI `model` field.

:::

## API Key

```python
import os

os.environ["HPC_AI_API_KEY"] = "your-api-key"
```

Optional: override the base URL (defaults to `https://api.hpc-ai.com/inference/v1`).

```python
os.environ["HPC_AI_API_BASE"] = "https://api.hpc-ai.com/inference/v1"
```

If you use another env name such as `HPC_AI_BASE_URL`, map it to `api_base` in your LiteLLM call or proxy `litellm_params`; LiteLLM reads `HPC_AI_API_BASE` by default.

## Sample Usage: Chat completion

```python
from litellm import completion
import os

os.environ["HPC_AI_API_KEY"] = "your-api-key"
response = completion(
    model="hpc_ai/minimax/minimax-m2.5",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=256,
)
print(response)
```

## Sample Usage: Streaming

```python
from litellm import completion
import os

os.environ["HPC_AI_API_KEY"] = "your-api-key"
response = completion(
    model="hpc_ai/moonshotai/kimi-k2.5",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True,
)

for chunk in response:
    print(chunk)
```

## Usage with LiteLLM Proxy Server

1. Add a model to your `config.yaml`:

```yaml
model_list:
  - model_name: hpc-ai-minimax
    litellm_params:
      model: hpc_ai/minimax/minimax-m2.5
      api_key: os.environ/HPC_AI_API_KEY
```

2. Start the proxy:

```bash
litellm --config /path/to/config.yaml
```

3. Send requests to the proxy using your alias (`hpc-ai-minimax` in the example above).

## Supported models (examples)

| LiteLLM model id | Notes |
| ---------------- | ----- |
| `hpc_ai/minimax/minimax-m2.5` | MiniMax M2.5 |
| `hpc_ai/moonshotai/kimi-k2.5` | Kimi K2.5 |

Pricing in `model_prices_and_context_window.json` may use placeholder token costs; set real rates when your billing API is available.
