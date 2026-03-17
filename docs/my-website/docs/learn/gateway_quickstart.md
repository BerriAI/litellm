---
title: Gateway Quickstart
sidebar_label: Gateway Quickstart
description: Start LiteLLM Gateway, add models and keys, then connect applications and SDKs to one shared endpoint.
---

import NavigationCards from '@site/src/components/NavigationCards';

Use this path if you need one shared OpenAI-compatible endpoint for a team or platform.

If you need a Docker or database-first setup, use the [Docker + Database tutorial](/docs/proxy/docker_quick_start). Otherwise, use the steps below to get to a working request fast.

## 1. Install The Gateway

```bash
pip install 'litellm[proxy]'
```

## 2. Set One Provider Key

```bash
export OPENAI_API_KEY="your-api-key"
```

## 3. Create `config.yaml`

```yaml
model_list:
  - model_name: gpt-4o-mini
    litellm_params:
      model: openai/gpt-4o-mini
      api_key: os.environ/OPENAI_API_KEY

general_settings:
  master_key: sk-1234
```

## 4. Start The Gateway

```bash
litellm --config config.yaml
```

You should see the proxy start on `http://0.0.0.0:4000`.

## 5. Send Your First Request

```bash
curl -X POST 'http://0.0.0.0:4000/chat/completions' \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer sk-1234' \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [
      {"role": "user", "content": "Hello from LiteLLM Gateway"}
    ]
  }'
```

## 6. Check The Response

If the request succeeds, the proxy returns `200 OK` with the same OpenAI-style response shape LiteLLM uses in the SDK.

The assistant text will be in:

```json
choices[0].message.content
```

It looks like this:

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1677858242,
  "model": "gpt-4o-mini",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! How can I help?"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 12,
    "completion_tokens": 9,
    "total_tokens": 21
  }
}
```

`id`, `created`, token counts, and message text will vary by request.

## 7. Add Keys And The UI

If you need virtual keys, spend tracking, or the admin UI, add a database next.

- Add `database_url` under `general_settings`
- Use [Virtual keys](/docs/proxy/virtual_keys) for key creation and budgets
- Use [Admin UI](/docs/proxy/ui) to manage models and keys
- Use the [Docker + Database tutorial](/docs/proxy/docker_quick_start) if you want a fuller setup

## 8. Pick Your Next Step

<NavigationCards
columns={3}
items={[
  {
    icon: "🎛️",
    title: "Model Config",
    description: "Add more models and gateway settings.",
    to: "/docs/proxy/configs",
  },
  {
    icon: "🔑",
    title: "Virtual Keys",
    description: "Create keys, budgets, and access controls.",
    to: "/docs/proxy/virtual_keys",
  },
  {
    icon: "📈",
    title: "Add Logging",
    description: "Capture logs, spend, and traces.",
    to: "/docs/proxy/logging",
  },
  {
    icon: "🔀",
    title: "Load Balance",
    description: "Route across deployments, regions, or providers.",
    to: "/docs/proxy/load_balancing",
  },
  {
    icon: "🛡️",
    title: "Add Guardrails",
    description: "Add safety checks and policy enforcement.",
    to: "/docs/proxy/guardrails/quick_start",
  },
  {
    icon: "🖥️",
    title: "Connect SDKs",
    description: "Point LiteLLM or OpenAI-compatible clients to the gateway.",
    to: "/docs/providers/litellm_proxy",
  },
  {
    icon: "📊",
    title: "Reliability",
    description: "Configure retries, fallbacks, and timeouts.",
    to: "/docs/proxy/reliability",
  },
]}
/>

## When To Use The SDK Path Instead

If you only need to call models from one application and do not need centralized auth or shared infrastructure, start with the [SDK Quickstart](/docs/learn/sdk_quickstart) instead.
