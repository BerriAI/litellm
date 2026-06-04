import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Neosantara

## Overview

| Property | Details |
|-------|-------|
| Description | Neosantara provides OpenAI-compatible access to chat and Responses API models. |
| Provider Route on LiteLLM | `neosantara/` |
| Link to Provider Doc | [Neosantara Documentation ↗](https://docs.neosantara.xyz) |
| Base URL | `https://api.neosantara.xyz/v1` |
| Supported Operations | [`/chat/completions`](#chat-completions) and [`/responses`](#responses-api) |

<br />
<br />

Use `neosantara/` as the model prefix when sending requests through LiteLLM.

## Required Variables

```python showLineNumbers title="Environment Variables"
os.environ["NEOSANTARA_API_KEY"] = ""  # your Neosantara API key
```

## Chat Completions

```python showLineNumbers title="Neosantara Chat Completion"
import os
from litellm import completion

os.environ["NEOSANTARA_API_KEY"] = ""  # your Neosantara API key

response = completion(
    model="neosantara/gemini-3-flash",
    messages=[{"role": "user", "content": "Halo, apa kabar?"}],
)

print(response)
```

## Responses API

```python showLineNumbers title="Neosantara Responses API"
import os
import litellm

os.environ["NEOSANTARA_API_KEY"] = ""  # your Neosantara API key

response = litellm.responses(
    model="neosantara/claude-4.5-sonnet",
    input="Halo, apa kabar?",
)

print(response)
```

## Usage - LiteLLM Proxy Server

```yaml showLineNumbers title="config.yaml"
model_list:
  - model_name: neosantara-chat
    litellm_params:
      model: neosantara/gemini-3-flash
      api_key: os.environ/NEOSANTARA_API_KEY
  - model_name: neosantara-responses
    litellm_params:
      model: neosantara/claude-4.5-sonnet
      api_key: os.environ/NEOSANTARA_API_KEY
```

## Custom API Base

```python showLineNumbers title="Custom API Base via env var"
import os
from litellm import completion

os.environ["NEOSANTARA_API_BASE"] = "https://api.neosantara.xyz/v1"
os.environ["NEOSANTARA_API_KEY"] = ""  # your API key

response = completion(
    model="neosantara/gemini-3-flash",
    messages=[{"role": "user", "content": "Hello!"}],
)
```
