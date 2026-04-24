---
slug: advisor-tool-chat-completions
title: "[Beta] Advisor Tool (SDK + Proxy)"
date: 2026-04-14T19:30:00
authors:
  - sameer
  - krrish
  - ishaan-alt
description: "Use Anthropic advisor-style orchestration across LiteLLM chat completions providers, including OpenAI, Azure, and Gemini."
tags: [advisor, anthropic, proxy, chat-completions, tools]
hide_table_of_contents: false
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Advisor Tool (Beta)

LiteLLM now supports the Anthropic advisor tool across `chat/completions` and `messages` APIs (SDK + proxy).

Use the advisor tool to let an executor model call a stronger advisor model during generation. For non-Anthropic providers, LiteLLM runs the advisor orchestration loop automatically. 

For updates after this post see the [latest Advisor Tool docs](/docs/completion/anthropic_advisor_tool).

:::info Beta

The advisor tool is in beta. LiteLLM adds the required `anthropic-beta: advisor-tool-2026-03-01` header automatically when it detects the advisor tool in your `tools` array.

:::

---

## Two tool formats

There are two ways to specify the advisor tool. Which one to use depends on your executor provider and setup.

### 1. Anthropic native format (`advisor_20260301`)

```json
{
  "type": "advisor_20260301",
  "name": "advisor",
  "model": "claude-opus-4-6"
}
```

The `model` field is **required** and specifies the advisor. Use this format when:
- Your executor is an Anthropic model and the advisor is `claude-opus-4-6` - Anthropic handles the advisor call natively, server-side.
- Your executor is any non-Anthropic model (OpenAI, Gemini, etc.) via the **Messages API** - LiteLLM's built-in interception converts this automatically.

### 2. OpenAI function format (`litellm_advisor`)

```json
{
  "type": "function",
  "function": {
    "name": "litellm_advisor",
    "description": "Consult a stronger advisor model.",
    "parameters": {
      "type": "object",
      "properties": { "question": { "type": "string" } },
      "required": ["question"]
    }
  }
}
```

This format does **not** carry a `model` field. The advisor model comes from your `AdvisorInterceptionLogger` setup or proxy config — see below. Use this format when calling through the **Chat Completions API** and you cannot send custom tool types (e.g. using a plain OpenAI client against the proxy).

:::warning You must configure the advisor model

Sending `litellm_advisor` as a bare function tool without setting up `AdvisorInterceptionLogger` (or the proxy `advisor_interception_params`) does nothing useful — the provider treats it as a regular custom tool and returns a `tool_use` response your code has to handle manually. Always pair it with the setup below.

:::

---

## Supported providers

| Provider | Chat Completions API | Messages API | Mode |
|----------|---------------------|--------------|------|
| **Anthropic** (executor + advisor = Opus 4.6) | ✅ | ✅ | Native server-side |
| **Anthropic** (executor) + **any other advisor** | ✅ | ✅ | LiteLLM orchestration loop |
| **OpenAI / Azure OpenAI** | ✅ | ✅ | LiteLLM orchestration loop |
| **Amazon Bedrock** | ✅ | ✅ | LiteLLM orchestration loop |
| **Google Vertex AI / Gemini** | ✅ | ✅ | LiteLLM orchestration loop |
| **Groq / Mistral / others** | ✅ | ✅ | LiteLLM orchestration loop |

**Native path:** Executor is Anthropic and advisor is `claude-opus-4-6` → Anthropic runs the advisor inference server-side. No LiteLLM orchestration involved.

**Orchestration path:** Everything else → LiteLLM intercepts the executor's tool call, runs the advisor as a sub-call using the credentials you configured, injects the advice, and continues. The advisor can be any provider.

---

## Chat Completions API

<Tabs>
<TabItem value="chat-completions-sdk" label="SDK">

### Configuring the advisor model (SDK)

Register `AdvisorInterceptionLogger` in `litellm.callbacks` and set `default_advisor_model`. This is what routes advisor sub-calls to the right model and credentials.

`default_advisor_model` is used when the tool definition has no `model` field (i.e. the `litellm_advisor` function format). If you pass the `advisor_20260301` native format with an explicit `model` field, that takes precedence.

```python showLineNumbers title="SDK setup — register AdvisorInterceptionLogger"
import asyncio
import litellm
from litellm.integrations.advisor_interception import AdvisorInterceptionLogger

litellm.callbacks = [
    AdvisorInterceptionLogger(
        default_advisor_model="openai/o3",
        enabled_providers=["anthropic", "openai"],
    )
]

async def main():
    response = await litellm.acompletion(
        model="openai/gpt-4o",
        messages=[
            {"role": "user", "content": "Build a concurrent worker pool in Go with graceful shutdown."}
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "litellm_advisor",
                    "description": "Consult a stronger advisor model.",
                    "parameters": {
                        "type": "object",
                        "properties": {"question": {"type": "string"}},
                        "required": ["question"],
                    },
                },
            }
        ],
        max_tokens=4096,
    )
    print(response.choices[0].message.content)

if __name__ == "__main__":
    asyncio.run(main())
```

You can also pass the advisor model directly in the tool definition using the native format — this overrides `default_advisor_model`:

```python showLineNumbers title="Advisor model set per-request in tool definition"
from litellm.integrations.advisor_interception import get_litellm_advisor_tool

tools=[
    get_litellm_advisor_tool(
        model="anthropic/claude-opus-4-6",  # overrides default_advisor_model for this request
        max_uses=2,
    )
]
```

---

### Streaming

```python showLineNumbers title="Streaming with Advisor Tool"
import asyncio
import litellm
from litellm.integrations.advisor_interception import AdvisorInterceptionLogger

litellm.callbacks = [
    AdvisorInterceptionLogger(default_advisor_model="openai/o3")
]

async def main():
    response = await litellm.acompletion(
        model="openai/gpt-4o-mini",
        messages=[
            {"role": "user", "content": "Implement a distributed rate limiter."}
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "litellm_advisor",
                    "description": "Consult a stronger advisor model.",
                    "parameters": {
                        "type": "object",
                        "properties": {"question": {"type": "string"}},
                        "required": ["question"],
                    },
                },
            }
        ],
        max_tokens=4096,
        stream=True,
    )

    async for chunk in response:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="")

asyncio.run(main())
```

:::note Streaming behavior

The advisor sub-inference does not stream. When the executor calls the advisor tool, the stream pauses, the advisor runs to completion, and its output is injected before the executor resumes streaming.

:::

</TabItem>
<TabItem value="chat-completions-proxy" label="Proxy">

### Configuring the advisor model (Proxy)

Add the advisor as a named deployment in `model_list` and reference it in `advisor_interception_params`. The proxy router resolves the correct credentials automatically.

```yaml showLineNumbers title="config.yaml"
model_list:
  # Advisor model — can be any provider
  - model_name: my-advisor
    litellm_params:
      model: openai/o3
      api_key: os.environ/OPENAI_API_KEY

  # Executor models
  - model_name: claude-sonnet
    litellm_params:
      model: anthropic/claude-sonnet-4-5
      api_key: os.environ/ANTHROPIC_API_KEY

  - model_name: gpt-4o-mini
    litellm_params:
      model: openai/gpt-4o-mini
      api_key: os.environ/OPENAI_API_KEY

  - model_name: gemini-flash
    litellm_params:
      model: vertex_ai/gemini-2.5-flash
      vertex_project: my-project
      vertex_location: us-central1

litellm_settings:
  callbacks: ["advisor_interception"]   # use callbacks, not success_callback
  advisor_interception_params:
    # Must be a model_name from model_list above.
    # The router uses this to pick the right deployment + credentials.
    default_advisor_model: "my-advisor"
```

:::info

- Use `callbacks`, not `success_callback`. The advisor hooks run through `litellm.callbacks`.
- `default_advisor_model` must match a `model_name` from `model_list`. This is how the proxy resolves the correct API key and deployment for the advisor sub-call.
- You can still override it per-request by passing `model` in an `advisor_20260301` tool definition.

:::

---

### Client request 
```python showLineNumbers title="Advisor via proxy"
from openai import OpenAI

client = OpenAI(
    api_key="your-litellm-proxy-key",
    base_url="http://0.0.0.0:4000/v1",
)

response = client.chat.completions.create(
    model="claude-sonnet",
    messages=[
        {"role": "user", "content": "Implement a distributed rate limiter in Python."}
    ],
    tools=[
        {
            "type": "function",
            "function": {
                "name": "litellm_advisor",
                "description": "Consult a stronger advisor model.",
                "parameters": {
                    "type": "object",
                    "properties": {"question": {"type": "string"}},
                    "required": ["question"],
                },
            },
        }
    ],
    max_tokens=4096,
)
print(response.choices[0].message.content)
```

### Response Structure


- **`advisor_tool_results`** : `server_tool_use` / `advisor_tool_result` pairs mirroring Anthropic’s shape (what the UI and logs can use).
- **`advisor_iterations`** : per-sub-call token usage (`message` vs `advisor_message`), aligned with aggregated `usage` on the completion.

```json title="chat.completion — gpt-4o-mini + claude-opus-4-6 advisor (excerpt)"
{
  "id": "chatcmpl-DVak9H9P1COkJNZMpz8IQWd4mqv7Y",
  "created": 1776421781,
  "model": "gpt-4o-mini",
  "object": "chat.completion",
  "system_fingerprint": "fp_2f65f9541c",
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,
      "message": {
        "content": "Here's the Python function that checks if a number is prime using the efficient approach as advised:\n\n```python\nimport math\n\ndef is_prime(n: int) -> bool:\n    \"\"\"…\"\"\"\n    …\n```\n\n### Explanation …\n\nFeel free to run this code in your Python environment to see the output!",
        "role": "assistant",
        "provider_specific_fields": {
          "refusal": null,
          "advisor_tool_results": [
            {
              "type": "server_tool_use",
              "id": "call_fxYpOvEL5PAUEpjNYw2PU99C",
              "name": "advisor"
            },
            {
              "type": "advisor_tool_result",
              "tool_use_id": "call_fxYpOvEL5PAUEpjNYw2PU99C",
              "content": {
                "type": "advisor_result",
                "text": "# Prime Number Checker in Python\n\n… full advisor markdown (implementation, tests, 6k±1 explanation) …"
              }
            }
          ],
          "advisor_iterations": [
            {
              "type": "message",
              "input_tokens": 113,
              "cache_read_input_tokens": 0,
              "cache_creation_input_tokens": 0,
              "output_tokens": 31
            },
            {
              "type": "advisor_message",
              "input_tokens": 47,
              "cache_read_input_tokens": 0,
              "cache_creation_input_tokens": 0,
              "output_tokens": 1024,
              "model": "claude-opus-4-6"
            },
            {
              "type": "message",
              "input_tokens": 1075,
              "cache_read_input_tokens": 0,
              "cache_creation_input_tokens": 0,
              "output_tokens": 649
            }
          ]
        },
        "annotations": []
      },
      "provider_specific_fields": {}
    }
  ],
  "usage": {
    "completion_tokens": 649,
    "prompt_tokens": 1075,
    "total_tokens": 1724,
    "completion_tokens_details": {
      "accepted_prediction_tokens": 0,
      "audio_tokens": 0,
      "reasoning_tokens": 0,
      "rejected_prediction_tokens": 0
    },
    "prompt_tokens_details": {
      "audio_tokens": 0,
      "cached_tokens": 0
    }
  },
  "service_tier": "default"
}
```

:::tip

Top-level `usage` reflects the **final** executor turn after injection. Use `advisor_iterations` for the full breakdown (executor → advisor → executor follow-up).

:::

</TabItem>
</Tabs>

---

## Messages API

Working demo: calling the LiteLLM proxy **`/v1/messages`** with the advisor tool (same flow as Anthropic’s Messages API).

<iframe width="840" height="500" src="https://www.loom.com/embed/ee10a816184042d09634fbd9dbfe1d79" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen></iframe>

<Tabs>
<TabItem value="messages-sdk" label="SDK">

The Messages API (`litellm.anthropic.messages`) has built-in interception — no callback registration needed. Pass the `advisor_20260301` tool with the `model` field and LiteLLM handles the rest.

#### Anthropic executor — native path

```python showLineNumbers title="Advisor Tool — Messages API, Anthropic native"
import asyncio
import litellm

async def main():
    response = await litellm.anthropic.messages.acreate(
        model="anthropic/claude-sonnet-4-6",
        messages=[
            {"role": "user", "content": "Build a concurrent worker pool in Go with graceful shutdown."}
        ],
        tools=[
            {
                "type": "advisor_20260301",
                "name": "advisor",
                "model": "claude-opus-4-6",   # Anthropic runs this natively
            }
        ],
        max_tokens=4096,
    )
    print(response)

asyncio.run(main())
```

#### Non-Anthropic executor — LiteLLM orchestration loop

When the executor is not Anthropic, or when the advisor model is not Claude Opus 4.6, LiteLLM runs the loop itself. The `model` field in the tool definition is the advisor — it can be any provider.

```python showLineNumbers title="Advisor Tool — Messages API, OpenAI executor"
import asyncio
import litellm

async def main():
    response = await litellm.anthropic.messages.acreate(
        model="openai/gpt-4o",
        messages=[
            {"role": "user", "content": "Implement a Python LRU cache with O(1) get and put."}
        ],
        tools=[
            {
                "type": "advisor_20260301",
                "name": "advisor",
                "model": "openai/o3",   # advisor model — any provider works
                "max_uses": 2,
            }
        ],
        max_tokens=1024,
    )
    print(response["content"][0]["text"])

asyncio.run(main())
```

#### Streaming

```python showLineNumbers title="Messages API Streaming with Advisor Tool"
import asyncio
import json
import litellm

async def main():
    response = await litellm.anthropic.messages.acreate(
        model="anthropic/claude-sonnet-4-6",
        messages=[
            {"role": "user", "content": "Implement a distributed rate limiter."}
        ],
        tools=[
            {
                "type": "advisor_20260301",
                "name": "advisor",
                "model": "claude-opus-4-6",
            }
        ],
        max_tokens=4096,
        stream=True,
    )

    async for chunk in response:
        if isinstance(chunk, bytes):
            for line in chunk.decode("utf-8").split("\n"):
                if line.startswith("data: "):
                    try:
                        print(json.loads(line[6:]))
                    except json.JSONDecodeError:
                        pass

asyncio.run(main())
```

</TabItem>
<TabItem value="messages-proxy" label="Proxy">

Use the same `config.yaml` shown in the Chat Completions proxy tab. The `advisor_interception_params` config applies to both APIs.

#### Client request — Anthropic SDK

```python showLineNumbers title="Advisor Tool via AI Gateway (Anthropic SDK)"
import anthropic

client = anthropic.Anthropic(
    api_key="your-litellm-proxy-key",
    base_url="http://0.0.0.0:4000",
)

response = client.beta.messages.create(
    model="claude-sonnet",
    max_tokens=4096,
    betas=["advisor-tool-2026-03-01"],
    messages=[
        {"role": "user", "content": "Build a concurrent worker pool in Go with graceful shutdown."}
    ],
    tools=[
        {
            "type": "advisor_20260301",
            "name": "advisor",
            "model": "my-advisor",   # model_name from config.yaml
        }
    ],
)
print(response)
```

</TabItem>
</Tabs>

---

## Response structure

### Messages API

Both native and orchestration paths return `server_tool_use` and `advisor_tool_result` blocks in the assistant content:

```json title="Messages API response"
{
  "id": "msg_6889286f94074716a142cf8edbc233c8",
  "type": "message",
  "role": "assistant",
  "model": "gpt-4o-mini",
  "stop_sequence": null,
  "usage": {
    "input_tokens": 1554,
    "output_tokens": 2659,
    "cache_read_input_tokens": 0,
    "cache_creation_input_tokens": 0,
    "iterations": [
      {
        "type": "message",
        "input_tokens": 90,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "output_tokens": 41
      },
      {
        "type": "advisor_message",
        "input_tokens": 37,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "output_tokens": 2494,
        "model": "gemini-3.1-pro-preview"
      },
      {
        "type": "message",
        "input_tokens": 1427,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "output_tokens": 124
      }
    ]
  },
  "content": [
    {
      "type": "text",
      "text": "To .."
    },
    {
      "type": "server_tool_use",
      "id": "call_vjS9XeTVVysRstjeG3prGJz5",
      "name": "advisor",
      "input": {}
    },
    {
      "type": "advisor_tool_result",
      "tool_use_id": "call_vjS9XeTVVysRstjeG3prGJz5",
      "content": {
        "type": "advisor_result",
        "text": "Welcome to the project room. As your Technical Advisor, I will help you conceptualize, design, and plan a mini router. … _(full advisor markdown in live responses: phases for use case, hardware, connectivity, OpenWrt stack, thermal/power, and next-step questions — truncated here for docs length.)_"
      }
    }
  ],
  "stop_reason": "end_turn",
  "_hidden_params": {
    "response_cost": 0.030328550000000003
  }
}
```

---

## Cost control

Advisor calls run as separate sub-inferences billed at the advisor model's rates. Usage is reported in `usage.iterations[]` in both messages endpoint and chat completion endpoint

```json title="Messages API usage with advisor sub-inference"
{
  "usage": {
    "input_tokens": 412,
    "output_tokens": 531,
    "iterations": [
      {
        "type": "message",
        "input_tokens": 412,
        "output_tokens": 89
      },
      {
        "type": "advisor_message",
        "model": "claude-opus-4-6",
        "input_tokens": 823,
        "output_tokens": 1612
      },
      {
        "type": "message",
        "input_tokens": 1348,
        "output_tokens": 442
      }
    ]
  }
}
```

Use `max_uses` in the tool definition to cap how many times the advisor can be called per request:

```python
tools=[
    {
        "type": "advisor_20260301",
        "name": "advisor",
        "model": "my-advisor",
        "max_uses": 2,   # raise AdvisorMaxIterationsError after 2 advisor calls
    }
]
```

---

## Logging

In the proxy **Logs** UI you can open a single orchestrated row and see the aggregated cost, token usage, and **cost breakdown** for advisor calls.

![LiteLLM proxy logs: advisor orchestration cost breakdown and usage](/img/blog/advisor-tool-logging-ui.png)

---

## Additional resources

- [Anthropic Advisor Tool Documentation](https://platform.claude.com/docs/en/agents-and-tools/tool-use/advisor-tool)
- [LiteLLM Tool Calling Guide](https://docs.litellm.ai/docs/completion/function_call)
