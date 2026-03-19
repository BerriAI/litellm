# Malachi

LiteLLM supports Malachi through its OpenAI-compatible API surface.

## Overview

| Property | Details |
|-------|-------|
| Description | Malachi is an OpenAI-compatible gateway that can expose chat and responses-style model endpoints. |
| Provider Route on LiteLLM | `malachi/` |
| Supported Operations | `/chat/completions`, `/responses` |

## API Base and API Key

```python
import os

os.environ["MALACHI_API_BASE"] = "https://your-malachi-host/v1"
os.environ["MALACHI_API_KEY"] = "your-api-key"  # optional if your gateway requires auth
```

## Usage

```python
from litellm import completion

response = completion(
    model="malachi/my-model",
    api_base="https://your-malachi-host/v1",
    api_key="your-api-key",
    messages=[{"role": "user", "content": "Hello from LiteLLM"}],
)

print(response)
```
