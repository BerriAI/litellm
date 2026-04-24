# Maritalk

| Property | Details |
|-------|-------|
| Description | Maritaca AI chat completions through LiteLLM's OpenAI-compatible provider path. |
| Provider Route on LiteLLM | `maritalk` |
| Default API Base | `https://chat.maritaca.ai/api` |
| Required API Key | `MARITALK_API_KEY` or `api_key=...` |
| Supported OpenAI Endpoints | `/chat/completions` |

## API Key

```python
import os

os.environ["MARITALK_API_KEY"] = "your-api-key"
os.environ["MARITALK_API_BASE"] = "https://chat.maritaca.ai/api"  # optional
```

## Sample Usage

```python
import os

from litellm import completion

os.environ["MARITALK_API_KEY"] = "your-api-key"

response = completion(
    model="maritalk",
    messages=[{"role": "user", "content": "Hello from LiteLLM"}],
)

print(response)
```

## Streaming

```python
from litellm import completion

response = completion(
    model="maritalk",
    messages=[{"role": "user", "content": "Stream a short answer"}],
    stream=True,
)

for chunk in response:
    print(chunk)
```

## Supported OpenAI Params

- `frequency_penalty`
- `presence_penalty`
- `top_p`
- `top_k`
- `temperature`
- `max_tokens`
- `n`
- `stop`
- `stream`
- `stream_options`
- `tools`
- `tool_choice`
