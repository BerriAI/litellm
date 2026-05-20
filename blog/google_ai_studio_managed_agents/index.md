---
slug: google-ai-studio-managed-agents
title: "Google AI Studio Managed Agents on LiteLLM"
date: 2026-05-19T10:00:00
authors:
  - sameer
  - krrish
  - ishaan
tags: [gemini, managed-agents, interactions, google-ai-studio]
hide_table_of_contents: false
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Google AI Studio - Managed Agents

LiteLLM now supports the [Google AI Studio Managed Agents API](https://ai.google.dev/gemini-api/docs/agents). Create, manage, and run custom agents through LiteLLM.

:::note
Available from LiteLLM `v1.87.0-dev.1` or above.
:::

{/* truncate */}

## Deploy this version

<Tabs>
<TabItem value="docker" label="Docker">

``` showLineNumbers title="docker run litellm"
docker run \
-e STORE_MODEL_IN_DB=True \
-p 4000:4000 \
ghcr.io/berriai/litellm:v1.87.0-dev.1
```

</TabItem>

<TabItem value="pip" label="Pip">

``` showLineNumbers title="pip install litellm"
pip install litellm==1.87.0.dev1
```

</TabItem>
</Tabs>

## Overview

There are two distinct steps:

1. **Create a custom agent**: /v1beta/agents defines your agent on the Gemini side (name, base model, instructions).
2. **Run the agent**: Once you have created a named agent, you can interact with it by specifying its resource name in the agent field of the /interactions request.

LiteLLM does **not** store the agent in its own database. The agent lives entirely on Google's side. LiteLLM is just the auth + routing layer.

## Quick start

<Tabs>
<TabItem value="proxy" label="Proxy">

Add your Gemini API key to the environment:

```bash
export GEMINI_API_KEY="AIzaSy..."
```

**Minimal `proxy_config.yaml`**:

```yaml
general_settings:
  master_key: "sk-1234"

environment_variables:
  GEMINI_API_KEY: "AIzaSy..."   # or set in shell env
```

Start the proxy:

```bash
litellm --config proxy_config.yaml
```

If `GEMINI_API_KEY` is not set, all managed-agent calls will fail with an auth error from Google.

</TabItem>
<TabItem value="sdk" label="SDK">

```python
import os
import litellm

os.environ["GEMINI_API_KEY"] = "AIzaSy..."
```

You can also pass `api_key="AIzaSy..."` to each call instead of setting the environment variable.

</TabItem>
</Tabs>

## 1. Create an agent
<Tabs>
<TabItem value="proxy" label="Proxy">

```bash
curl -X POST "http://localhost:4000/v1beta/agents" \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my-custom-slides-agent",
    "base_agent": "antigravity-preview-05-2026",
    "instructions": "You are a helpful assistant that creates slides.",
    "base_environment": {"env_id": "YOUR_ENVIRONMENT_ID"}
}'
```

**Response:**

```json
{
  "id": "my-slides-agent",
  "base_agent": "antigravity-preview-05-2026",
  "system_instruction": "You are a helpful assistant that creates slides."
}
```

</TabItem>
<TabItem value="sdk" label="SDK">

```python
response = litellm.interactions.agents.create(
    name="my-slides-agent",
    base_agent="antigravity-preview-05-2026",
    instructions="You are a helpful assistant that creates slides.",
    custom_llm_provider="gemini",
    base_environment={"env_id": "YOUR_ENVIRONMENT_ID"}
)
print(response.id)  # "my-slides-agent"
```

Async variant: `litellm.interactions.agents.acreate(...)`.

</TabItem>
</Tabs>

**Parameters:**

| Field | Required | Description |
|---|---|---|
| `name` | Yes | Unique agent identifier, used as the ID in later calls |
| `base_agent` | Yes | Base model to build on. Currently only `"antigravity-preview-05-2026"` is supported by Google |
| `instructions` | No | System-level instructions for the agent |
| `base_environment` | No | Environment config (e.g. GCS skill sources) |

> Calling create twice with the same `name` returns a `409 Conflict` from Google.

## 2. Run an agent

<Tabs>
<TabItem value="proxy" label="Proxy">

```bash
curl -X POST "http://localhost:4000/v1beta/interactions" \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "agent": "my-slides-agent",
    "input": "Create a slide deck on the Fibonacci sequence",
    "environment": "remote" # required for agents
  }'
```

</TabItem>
<TabItem value="sdk" label="SDK">

```python
response = litellm.interactions.create(
    agent="my-slides-agent",
    input="Create a slide deck on the Fibonacci sequence",
    environment="remote"
)
print(response)
```

Async variant: `litellm.interactions.acreate(...)`.

</TabItem>
</Tabs>

Note: pass `agent`, **not** `model`. The agent name is not a LiteLLM model, do not put it in the `model` field.

See also: [/interactions](/docs/interactions) for the full Interactions API.

## Manage agents

### List agents

<Tabs>
<TabItem value="proxy" label="Proxy">

```bash
curl "http://localhost:4000/v1beta/agents" \
  -H "Authorization: Bearer sk-1234"
```

**Response**
```json
{
    "agents": [
        {
            "id": "my-custom-slides-agent"
        },
        {
            "id": "my-custom-slides-agent-1"
        }
    ]
}
```

</TabItem>
<TabItem value="sdk" label="SDK">

```python
agents = litellm.interactions.agents.list()
```

</TabItem>
</Tabs>

### Get an agent

<Tabs>
<TabItem value="proxy" label="Proxy">

```bash
curl "http://localhost:4000/v1beta/agents/my-slides-agent" \
  -H "Authorization: Bearer sk-1234"
```

**Response**
```json
{
    "id": "my-custom-slides-agent",
    "base_agent": "antigravity-preview-05-2026",
    "system_instruction": "You are a helpful assistant that creates slides.",
    "base_environment": {
        "sources": [
            {
                "type": "gcs",
                "source": "gs://eap-templates/slides-skill",
                "target": "/.agents/skills/slides-skill"
            }
        ],
        "type": "remote"
    }
}
```

</TabItem>
<TabItem value="sdk" label="SDK">

```python
agent = litellm.interactions.agents.get(
    name="my-slides-agent"
)
```

</TabItem>
</Tabs>

### Delete an agent

<Tabs>
<TabItem value="proxy" label="Proxy">

```bash
curl -X DELETE "http://localhost:4000/v1beta/agents/my-slides-agent" \
  -H "Authorization: Bearer sk-1234"
```

</TabItem>
<TabItem value="sdk" label="SDK">

```python
litellm.interactions.agents.delete(
    name="my-slides-agent",
    custom_llm_provider="gemini",
)
```

</TabItem>
</Tabs>

### List agent versions

<Tabs>
<TabItem value="proxy" label="Proxy">

```bash
curl "http://localhost:4000/v1beta/agents/my-slides-agent/versions" \
  -H "Authorization: Bearer sk-1234"
```
**Response**
```json
{
    "agentVersions": [
        {
            "agent": "antigravity-preview-05-2026",
            "base_environment": {
                "env_id": "sdsdd"
            },
            "instructions": "You are a helpful assistant that creates slides",
            "name": "agents/my-custom-slides-agent-1/versions/a7616fd3-4e3e-48e7-aea7-9ac76b4f37ab"
        }
    ]
}
```


</TabItem>
<TabItem value="sdk" label="SDK">

```python
versions = litellm.interactions.agents.list_versions(
    name="my-slides-agent",
    custom_llm_provider="gemini",
)
```

</TabItem>
</Tabs>

## Authentication

| Method | How to provide the key |
|---|---|
| **Proxy** | Set `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) in the proxy's environment. Virtual keys (`sk-...`) authenticate users *to the proxy*; the proxy uses your Gemini key to talk to Google. |
| **SDK** | Set `GEMINI_API_KEY` in the environment, or pass `api_key="AIzaSy..."` to each call. |

There is no way to use managed agents with any provider other than Google AI Studio. Other providers are not supported by this API.


## Limitations

- `base_agent` only accepts `"antigravity-preview-05-2026"` (Google's current restriction).
- Agents are stored on Google's side only. LiteLLM does not persist them in its database. If you delete an agent via Google's API directly, the proxy will not know.
- Using the Interactions API via the `agent` param is only supported by Gemini as of now. Use the `model` param to call other providers' models.
- `GEMINI_API_KEY` / `GOOGLE_API_KEY` must be present in the proxy environment. Passing the key per-request via `api_key` is supported in the SDK but not currently via the proxy endpoint.