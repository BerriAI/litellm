---
id: index
title: Getting Started
sidebar_label: Quickstart
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import NavigationCards from '@site/src/components/NavigationCards';
import Image from '@theme/IdealImage';

<Image style={{padding: '10px', margin: '0 0 2.5rem'}} img={require('../img/hero.png')} />

**LiteLLM** is an open-source library that gives you a single, unified interface to call 100+ LLMs — OpenAI, Anthropic, Vertex AI, Bedrock, and more — using the OpenAI format.

- Call any provider using the same `completion()` interface — no re-learning the API for each one
- Consistent output format regardless of which provider or model you use
- Built-in retry / fallback logic across multiple deployments via the [Router](./routing.md)
- Self-hosted [LLM Gateway (Proxy)](./simple_proxy) with virtual keys, cost tracking, and an admin UI

[![PyPI](https://img.shields.io/pypi/v/litellm.svg)](https://pypi.org/project/litellm/)
[![GitHub Stars](https://img.shields.io/github/stars/BerriAI/litellm?style=social)](https://github.com/BerriAI/litellm)

---

## Installation

```shell
pip install litellm
```

To run the full Proxy Server (LLM Gateway):

```shell
pip install 'litellm[proxy]'
```

---

## Quick Start

Make your first LLM call using the provider of your choice:

<Tabs>
<TabItem value="openai" label="OpenAI">

```python
from litellm import completion
import os

os.environ["OPENAI_API_KEY"] = "your-api-key"

response = completion(
  model="openai/gpt-4o",
  messages=[{"role": "user", "content": "Hello, how are you?"}]
)
print(response.choices[0].message.content)
```

</TabItem>
<TabItem value="anthropic" label="Anthropic">

```python
from litellm import completion
import os

os.environ["ANTHROPIC_API_KEY"] = "your-api-key"

response = completion(
  model="anthropic/claude-3-5-sonnet-20241022",
  messages=[{"role": "user", "content": "Hello, how are you?"}]
)
print(response.choices[0].message.content)
```

</TabItem>
<TabItem value="vertex" label="Vertex AI">

```python
from litellm import completion
import os

# auth: run 'gcloud auth application-default login'
os.environ["VERTEXAI_PROJECT"] = "your-project-id"
os.environ["VERTEXAI_LOCATION"] = "us-central1"

response = completion(
  model="vertex_ai/gemini-1.5-pro",
  messages=[{"role": "user", "content": "Hello, how are you?"}]
)
print(response.choices[0].message.content)
```

</TabItem>
<TabItem value="bedrock" label="Bedrock">

```python
from litellm import completion
import os

os.environ["AWS_ACCESS_KEY_ID"] = "your-key"
os.environ["AWS_SECRET_ACCESS_KEY"] = "your-secret"
os.environ["AWS_REGION_NAME"] = "us-east-1"

response = completion(
  model="bedrock/anthropic.claude-haiku-4-5-20251001:0",
  messages=[{"role": "user", "content": "Hello, how are you?"}]
)
print(response.choices[0].message.content)
```

</TabItem>
<TabItem value="ollama" label="Ollama">

```python
from litellm import completion

response = completion(
  model="ollama/llama3",
  messages=[{"role": "user", "content": "Hello, how are you?"}],
  api_base="http://localhost:11434"
)
print(response.choices[0].message.content)
```

</TabItem>
<TabItem value="azure" label="Azure OpenAI">

```python
from litellm import completion
import os

os.environ["AZURE_API_KEY"] = "your-key"
os.environ["AZURE_API_BASE"] = "https://your-resource.openai.azure.com"
os.environ["AZURE_API_VERSION"] = "2024-02-01"

response = completion(
  model="azure/your-deployment-name",
  messages=[{"role": "user", "content": "Hello, how are you?"}]
)
print(response.choices[0].message.content)
```

</TabItem>
</Tabs>

Every response follows the OpenAI Chat Completions format, regardless of provider. ✅

### Response Format

Non-streaming responses return a `ModelResponse` object:

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1677858242,
  "model": "gpt-4o",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! I'm doing well, thanks for asking."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 13,
    "completion_tokens": 12,
    "total_tokens": 25
  }
}
```

Streaming responses (`stream=True`) yield `ModelResponseStream` chunks:

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion.chunk",
  "created": 1677858242,
  "model": "gpt-4o",
  "choices": [
    {
      "index": 0,
      "delta": {
        "role": "assistant",
        "content": "Hello"
      },
      "finish_reason": null
    }
  ]
}
```

📖 [Full output format reference →](./completion/output)

:::tip Open in Colab
<a target="_blank" href="https://colab.research.google.com/github/BerriAI/litellm/blob/main/cookbook/liteLLM_Getting_Started.ipynb">
<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
:::

---

## New to LiteLLM?

**Want to get started fast?** Head to [Tutorials](/docs/tutorials) for step-by-step walkthroughs — AI coding tools, agent SDKs, proxy setup, and more.

**Need to understand a specific feature?** Check [Guides](/docs/guides) for streaming, function calling, prompt caching, and other how-tos.

---

## Choose Your Path

<NavigationCards
columns={2}
items={[
{
icon: "🐍",
title: "Python SDK",
description: "Integrate LiteLLM directly into your Python application. Drop-in replacement for the OpenAI client.",
listDescription: [
"completion(), embedding(), image_generation() and more",
"Router with retry, fallback, and load balancing",
"OpenAI-compatible exceptions across all providers",
"Observability callbacks (Langfuse, MLflow, Helicone…)",
],
to: "#litellm-python-sdk",
},
{
icon: "🖥️",
title: "Proxy Server (LLM Gateway)",
description: "Self-hosted gateway for platform teams managing LLM access across an organization.",
listDescription: [
"Virtual keys with per-key/team/user budgets",
"Centralized logging, guardrails, and caching",
"Admin UI for monitoring and management",
"Drop-in replacement for any OpenAI-compatible client",
],
to: "#litellm-proxy-server-llm-gateway",
},
]}
/>

---

## LiteLLM Python SDK

### Streaming

Add `stream=True` to receive chunks as they are generated:

```python
from litellm import completion
import os

os.environ["OPENAI_API_KEY"] = "your-api-key"

for chunk in completion(
  model="openai/gpt-4o",
  messages=[{"role": "user", "content": "Write a short poem"}],
  stream=True,
):
    print(chunk.choices[0].delta.content or "", end="")
```

### Exception Handling

LiteLLM maps every provider's errors to the OpenAI exception types — your existing error handling works out of the box:

```python
import litellm

try:
    litellm.completion(
      model="anthropic/claude-instant-1",
      messages=[{"role": "user", "content": "Hey!"}]
    )
except litellm.AuthenticationError as e:
    print(f"Bad API key: {e}")
except litellm.RateLimitError as e:
    print(f"Rate limited: {e}")
except litellm.APIError as e:
    print(f"API error: {e}")
```

### Logging & Observability

Send input/output to Langfuse, MLflow, Helicone, Lunary, and more with a single line:

```python
import litellm

litellm.success_callback = ["langfuse", "mlflow", "helicone"]

response = litellm.completion(
  model="gpt-4o",
  messages=[{"role": "user", "content": "Hi!"}]
)
```

📖 [See all observability integrations →](/docs/observability/agentops_integration)

### Track Costs & Usage

Use a callback to capture cost per response:

```python
import litellm

def track_cost(kwargs, completion_response, start_time, end_time):
    print("Cost:", kwargs.get("response_cost", 0))

litellm.success_callback = [track_cost]

litellm.completion(
  model="gpt-4o",
  messages=[{"role": "user", "content": "Hello!"}],
  stream=True
)
```

📖 [Custom callback docs →](./observability/custom_callback)

---

## LiteLLM Proxy Server (LLM Gateway)

The proxy is a self-hosted OpenAI-compatible gateway. Any client that works with OpenAI works with the proxy — no code changes needed.

![LiteLLM Proxy Dashboard](https://github.com/BerriAI/litellm/assets/29436595/47c97d5e-b9be-4839-b28c-43d7f4f10033)

#### Step 1 — Start the proxy

<Tabs>
<TabItem value="pip" label="pip">

```shell
litellm --model huggingface/bigcode/starcoder
# Proxy running on http://0.0.0.0:4000
```

</TabItem>
<TabItem value="docker" label="Docker">

```yaml title="litellm_config.yaml"
model_list:
  - model_name: gpt-3.5-turbo
    litellm_params:
      model: azure/your-deployment
      api_base: os.environ/AZURE_API_BASE
      api_key: os.environ/AZURE_API_KEY
      api_version: "2023-07-01-preview"
```

```shell
docker run \
  -v $(pwd)/litellm_config.yaml:/app/config.yaml \
  -e AZURE_API_KEY=your-key \
  -e AZURE_API_BASE=https://your-resource.openai.azure.com/ \
  -p 4000:4000 \
  docker.litellm.ai/berriai/litellm:main-latest \
  --config /app/config.yaml --detailed_debug
```

</TabItem>
</Tabs>

#### Step 2 — Call it with the OpenAI client

```python
import openai

client = openai.OpenAI(api_key="anything", base_url="http://0.0.0.0:4000")

response = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[{"role": "user", "content": "Write a short poem"}]
)
print(response.choices[0].message.content)
```

👉 [Full proxy quickstart with Docker →](./proxy/docker_quick_start)

:::tip Debugging tool
Use [**`/utils/transform_request`**](./utils/transform_request) to inspect exactly what LiteLLM sends to any provider — useful for debugging prompt formatting, header issues, and provider-specific parameters.
:::

🔗 [Interactive API explorer (Swagger) →](https://litellm-api.up.railway.app/)

---

## Agent & MCP Gateway

LiteLLM is a unified gateway for **LLMs, agents, and MCP** — you don't need a separate agent or MCP gateway. One endpoint for 100+ models, A2A agents, and MCP tools.

<NavigationCards
columns={2}
items={[
{
icon: "🔗",
title: "A2A Agents",
description: "Add and invoke A2A agents via the LiteLLM gateway.",
to: "/docs/a2a",
},
{
icon: "🛠️",
title: "MCP Gateway",
description: "Central MCP endpoint with per-key access control.",
to: "/docs/mcp",
},
]}
/>

---

## What to Explore Next

<NavigationCards
columns={3}
items={[
{
icon: "🔀",
title: "Routing & Load Balancing",
description: "Load balance across deployments and set automatic fallbacks.",
to: "/docs/routing-load-balancing",
},
{
icon: "🔑",
title: "Virtual Keys",
description: "Manage access, budgets, and rate limits per team or user.",
to: "/docs/proxy/virtual_keys",
},
{
icon: "📊",
title: "Spend Tracking",
description: "Track costs per key, team, and user across all providers.",
to: "/docs/proxy/cost_tracking",
},
{
icon: "🛡️",
title: "Guardrails",
description: "Add content filtering, PII masking, and safety checks.",
to: "/docs/proxy/guardrails/quick_start",
},
{
icon: "📡",
title: "Observability",
description: "Integrate with Langfuse, MLflow, Helicone, and more.",
to: "/docs/observability/agentops_integration",
},
{
icon: "🏭",
title: "Enterprise",
description: "SSO/SAML, audit logs, and advanced security for production.",
to: "/docs/enterprise",
},
]}
/>
