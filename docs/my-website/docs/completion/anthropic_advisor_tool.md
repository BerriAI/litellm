import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Advisor Tool

LiteLLM now supports the Anthropic advisor tool across `chat/completions` and `messages` APIs (SDK + proxy).

Use the advisor tool to let an executor model call a stronger advisor model during generation. For non-Anthropic providers, LiteLLM runs the advisor orchestration loop automatically.

:::info Beta

The advisor tool is in beta. Include `anthropic-beta: advisor-tool-2026-03-01` in your requests — LiteLLM adds this automatically when it detects the advisor tool in your `tools` array.

:::

## Supported Providers

| Provider | Chat Completions API | Messages API | Notes |
|----------|---------------------|--------------|-------|
| **Anthropic API** | ✅ | ✅ | Native — runs server-side |
| **OpenAI / Azure OpenAI** | ✅ | ✅ | LiteLLM orchestration loop |
| **Amazon Bedrock** | ✅ | ✅ | LiteLLM orchestration loop |
| **Google Vertex AI** | ✅ | ✅ | LiteLLM orchestration loop |
| **Groq / Mistral / others** | ✅ | ✅ | LiteLLM orchestration loop |

For non-Anthropic providers, LiteLLM implements the advisor loop itself.

- **Messages API** (`litellm.anthropic.messages.create/acreate`): built-in interception in the messages handler
- **Chat Completions API** (`litellm.completion/acompletion`): enable `AdvisorInterceptionLogger` to convert advisor tools + run the loop

When a request arrives with an `advisor_20260301` tool and a non-Anthropic provider, LiteLLM translates the advisor tool into a regular function tool the provider understands, then runs an orchestration loop:

![Advisor Orchestration Flow](/img/advisor_orchestration_flow.svg)

**What LiteLLM does for you:**

- Strips `advisor_20260301` from the outgoing request — the provider only sees a standard function tool named `advisor`
- When the executor calls it, intercepts before the result reaches you, runs the advisor sub-call, and injects the advice
- Strips any `advisor_tool_result` / `server_tool_use` blocks from message history on re-send so non-Anthropic providers never see Anthropic-specific types
- Wraps the final response in an SSE stream if you requested `stream=True`
- Enforces `max_uses` as a hard cap — `AdvisorMaxIterationsError` is raised if exceeded; `max_uses=0` disables the advisor entirely

## Model Compatibility

The advisor model is fully configurable. You can use any model deployed in your proxy as the advisor — it does not need to be Anthropic.

For **Anthropic-native** requests (where Anthropic runs the advisor server-side), the executor and advisor must form a valid Anthropic pair:

| Executor | Advisor |
|----------|---------|
| `claude-haiku-4-5-20251001` | `claude-opus-4-6` |
| `claude-sonnet-4-6` | `claude-opus-4-6` |
| `claude-opus-4-6` | `claude-opus-4-6` |

For **non-Anthropic** executors (where LiteLLM orchestrates the advisor loop), you can use any model as the advisor — including OpenAI, Vertex AI, Bedrock, etc.

---

## Chat Completions API

<Tabs>
<TabItem value="chat-completions-sdk" label="SDK">

#### Basic Example (Anthropic-native executor)

```python showLineNumbers title="Advisor Tool — litellm.completion()"
import litellm

response = litellm.completion(
    model="anthropic/claude-sonnet-4-6",
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
                    "properties": {
                        "question": {"type": "string"}
                    },
                    "required": ["question"],
                },
            },
        }
    ],
    max_tokens=4096,
)

print(response.choices[0].message.content)
```

#### Non-Anthropic Executor (Chat Completions interception)

```python showLineNumbers title="Advisor Tool with OpenAI executor via chat-completions"
import asyncio
import litellm
from litellm.integrations.advisor_interception import (
    AdvisorInterceptionLogger,
    get_litellm_advisor_tool,
)

litellm.callbacks = [AdvisorInterceptionLogger(enabled_providers=["openai"])]

async def main():
    response = await litellm.acompletion(
        model="gpt-5.4-mini",
        messages=[
            {"role": "user", "content": "Build a concurrent worker pool in Go with graceful shutdown."}
        ],
        # You can still use Anthropic-native advisor tool format.
        tools=[get_litellm_advisor_tool(model="claude-opus-4-6")],
        max_tokens=4096,
    )
    print(response.choices[0].message.content)

asyncio.run(main())
```

::::note

`AdvisorInterceptionLogger` converts advisor tool definitions to provider-compatible function tools for non-Anthropic chat-completions providers and runs the advisor sub-call loop server-side.

::::

#### With Optional Parameters

```python showLineNumbers title="Advisor Tool with max_uses and caching"
import litellm

response = litellm.completion(
    model="anthropic/claude-sonnet-4-6",
    messages=[
        {"role": "user", "content": "Build a REST API with authentication in Python."}
    ],
    tools=[
        {
            "type": "advisor_20260301",
            "name": "advisor",
            "model": "claude-opus-4-6",
            "max_uses": 3,                             # cap advisor calls per request
            "caching": {"type": "ephemeral", "ttl": "5m"},  # enable for 3+ calls per conversation
        }
    ],
    max_tokens=4096,
)
```

#### Streaming

```python showLineNumbers title="Streaming with Advisor Tool"
import litellm

response = litellm.completion(
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

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

:::note Streaming behavior

The advisor sub-inference does not stream. The executor's stream pauses while the advisor runs, then the full advisor result arrives in a single event. Executor output resumes streaming afterward.

:::

#### Multi-Turn Conversation

```python showLineNumbers title="Multi-Turn with Advisor Tool"
import litellm

tools = [
    {
        "type": "advisor_20260301",
        "name": "advisor",
        "model": "claude-opus-4-6",
    }
]

messages = [
    {"role": "user", "content": "Build a concurrent worker pool in Go with graceful shutdown."}
]

response = litellm.completion(
    model="anthropic/claude-sonnet-4-6",
    messages=messages,
    tools=tools,
    max_tokens=4096,
)

# Append the full response (includes server_tool_use + advisor_tool_result blocks)
messages.append({"role": "assistant", "content": response.choices[0].message.content})

# Continue the conversation — keep the same tools array
messages.append({"role": "user", "content": "Now add a max-in-flight limit of 10."})

response2 = litellm.completion(
    model="anthropic/claude-sonnet-4-6",
    messages=messages,
    tools=tools,
    max_tokens=4096,
)
```

:::tip Auto-strip on follow-up turns

LiteLLM automatically strips `advisor_tool_result` blocks from message history when the advisor tool is not present in the current request. This prevents the Anthropic 400 error that would otherwise occur.

:::

</TabItem>
<TabItem value="chat-completions-proxy" label="Proxy">

#### Proxy Configuration

Configure the advisor model as a deployment in your `model_list` and reference it in `advisor_interception_params`. This ensures the advisor sub-calls use the correct credentials and go through the proxy's deployment routing.

```yaml showLineNumbers title="config.yaml"
model_list:
  # The advisor model
  - model_name: advisor-model
    litellm_params:
      model: anthropic/claude-sonnet-4-20250514
      api_key: os.environ/ANTHROPIC_API_KEY

  # Executor models
  - model_name: claude-sonnet
    litellm_params:
      model: anthropic/claude-sonnet-4-6
      api_key: os.environ/ANTHROPIC_API_KEY

  - model_name: gemini-flash
    litellm_params:
      model: vertex_ai/gemini-2.5-flash
      vertex_project: my-project
      vertex_location: us-central1

litellm_settings:
  callbacks: ["advisor_interception"]
  advisor_interception_params:
    # Must match a model_name from model_list — the router resolves
    # the correct deployment and credentials automatically.
    default_advisor_model: "advisor-model"
```

:::info Important

- Use `callbacks`, not `success_callback`. The advisor interception hooks run through `litellm.callbacks`.
- The `default_advisor_model` value must be a `model_name` from your `model_list`. The proxy router resolves it to the correct deployment with the correct API key. This means you can use any provider as your advisor model — not just Anthropic.

:::

#### Client Request via Proxy

```python showLineNumbers title="Advisor Tool via AI Gateway"
from openai import OpenAI

client = OpenAI(
    api_key="your-litellm-proxy-key",
    base_url="http://0.0.0.0:4000/v1"
)

response = client.chat.completions.create(
    model="claude-sonnet",
    messages=[
        {"role": "user", "content": "Implement a distributed rate limiter in Python."}
    ],
    tools=[
        {
            "type": "advisor_20260301",
            "name": "advisor",
            "model": "advisor-model",
        }
    ],
    max_tokens=4096,
)
```

#### Client Request via Proxy (OpenAI-compatible function tool)

Use this format when your chat-completions client sends OpenAI-style tools. The proxy uses the `default_advisor_model` from your config.

```python showLineNumbers title="Proxy Chat Completions with litellm_advisor"
from openai import OpenAI

client = OpenAI(
    api_key="your-litellm-proxy-key",
    base_url="http://0.0.0.0:4000/v1",
)

response = client.chat.completions.create(
    model="gemini-flash",
    messages=[
        {"role": "user", "content": "Call advisor once, then answer in one line: integration ok."}
    ],
    tools=[
        {
            "type": "function",
            "function": {
                "name": "litellm_advisor",
                "description": "Consult a stronger advisor model.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"}
                    },
                    "required": ["question"],
                },
            },
        }
    ],
    max_tokens=512,
)
print(response.choices[0].message.content)
```

::::note

For non-Anthropic chat-completions providers behind proxy, this OpenAI-compatible
`litellm_advisor` function tool is the recommended request shape.
The advisor model is determined by the `default_advisor_model` in your `advisor_interception_params` config.

::::

</TabItem>
</Tabs>

---

## Messages API

<Tabs>
<TabItem value="messages-sdk" label="SDK">

#### Basic Example

```python showLineNumbers title="Advisor Tool — litellm.anthropic.messages"
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
                "model": "claude-opus-4-6",
            }
        ],
        max_tokens=4096,
    )
    print(response)

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

#### Proxy Configuration

Use the same config shown in the Chat Completions proxy tab. The `advisor_interception_params` config applies to both APIs.

```yaml showLineNumbers title="config.yaml"
model_list:
  - model_name: advisor-model
    litellm_params:
      model: anthropic/claude-sonnet-4-20250514
      api_key: os.environ/ANTHROPIC_API_KEY
  - model_name: claude-sonnet
    litellm_params:
      model: anthropic/claude-sonnet-4-6
      api_key: os.environ/ANTHROPIC_API_KEY

litellm_settings:
  callbacks: ["advisor_interception"]
  advisor_interception_params:
    default_advisor_model: "advisor-model"
```

#### Client Request via Proxy (Anthropic SDK)

```python showLineNumbers title="Advisor Tool via AI Gateway (Anthropic SDK)"
import anthropic

client = anthropic.Anthropic(
    api_key="your-litellm-proxy-key",
    base_url="http://0.0.0.0:4000"
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
            "model": "advisor-model",
        }
    ],
)
print(response)
```

#### Non-Anthropic Provider (LiteLLM orchestration loop)

```python showLineNumbers title="Advisor Tool with OpenAI executor"
import asyncio
import litellm

async def main():
    # executor: openai/gpt-4.1-mini  |  advisor: claude-opus-4-6
    # LiteLLM runs the orchestration loop automatically
    response = await litellm.anthropic.messages.acreate(
        model="openai/gpt-4.1-mini",
        messages=[
            {"role": "user", "content": "Implement a Python LRU cache with O(1) get and put."}
        ],
        tools=[
            {
                "type": "advisor_20260301",
                "name": "advisor",
                "model": "claude-opus-4-6",
                "max_uses": 3,
            }
        ],
        max_tokens=1024,
        custom_llm_provider="openai",
    )
    # Final response is clean — no advisor tool_use blocks
    print(response["content"][0]["text"])

asyncio.run(main())
```

</TabItem>
</Tabs>

---

## Response Structure

A successful advisor call returns `server_tool_use` and `advisor_tool_result` blocks in the assistant content:

```json title="Response with advisor blocks"
{
  "role": "assistant",
  "content": [
    {
      "type": "text",
      "text": "Let me consult the advisor on this."
    },
    {
      "type": "server_tool_use",
      "id": "srvtoolu_abc123",
      "name": "advisor",
      "input": {}
    },
    {
      "type": "advisor_tool_result",
      "tool_use_id": "srvtoolu_abc123",
      "content": {
        "type": "advisor_result",
        "text": "Use a channel-based coordination pattern. The tricky part is draining in-flight work during shutdown: close the input channel first, then wait on a WaitGroup..."
      }
    },
    {
      "type": "text",
      "text": "Here's the implementation using a channel-based coordination pattern..."
    }
  ]
}
```

Pass the full assistant content, including advisor blocks, back on subsequent turns. LiteLLM handles this automatically through `provider_specific_fields`.

---

## Cost Control

Advisor calls run as a separate sub-inference billed at the advisor model's rates. Usage is reported in `usage.iterations[]`:

```json title="Usage with advisor sub-inference"
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

Top-level `usage` reflects executor tokens only. Advisor tokens appear in `iterations` entries with `type: "advisor_message"` and are billed at Opus rates.

## Additional Resources

- [Anthropic Advisor Tool Documentation](https://platform.claude.com/docs/en/agents-and-tools/tool-use/advisor-tool)
- [LiteLLM Tool Calling Guide](https://docs.litellm.ai/docs/completion/function_call)
