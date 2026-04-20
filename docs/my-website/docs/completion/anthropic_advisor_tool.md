import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Advisor Tool (Beta)

LiteLLM supports the Anthropic advisor tool across `chat/completions` and `messages` APIs (SDK + proxy).

Use the advisor tool to let an executor model call a stronger advisor model during generation. For non-Anthropic providers, LiteLLM runs the advisor orchestration loop automatically.

:::info Beta

The advisor tool is in beta. LiteLLM adds the required `anthropic-beta: advisor-tool-2026-03-01` header automatically when it detects the advisor tool in your `tools` array.

:::

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

- Your executor is an Anthropic model and the advisor is `claude-opus-4-6` — Anthropic handles the advisor call natively, server-side.
- Your executor is any non-Anthropic model (OpenAI, Gemini, etc.) via the **Messages API** — LiteLLM's built-in interception converts this automatically.

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

## Model compatibility

The advisor model is fully configurable for orchestrated (non-native) paths — use any model deployed in your proxy.

For **Anthropic-native** requests (Anthropic runs the advisor server-side), the executor and advisor must form a valid Anthropic pair:

| Executor | Advisor |
|----------|---------|
| `claude-haiku-4-5-20251001` | `claude-opus-4-6` |
| `claude-sonnet-4-6` | `claude-opus-4-6` |
| `claude-opus-4-6` | `claude-opus-4-6` |

For **non-Anthropic** executors (LiteLLM orchestrates the advisor loop), you can use any model as the advisor — OpenAI, Vertex AI, Bedrock, etc.

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

#### Anthropic-native executor (`advisor_20260301`)

```python showLineNumbers title="Advisor Tool — litellm.completion() with native tool type"
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
            "max_uses": 3,
            "caching": {"type": "ephemeral", "ttl": "5m"},
        }
    ],
    max_tokens=4096,
)
print(response.choices[0].message.content)
```

#### Multi-turn conversation

```python showLineNumbers title="Multi-turn with advisor tool"
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

messages.append({"role": "assistant", "content": response.choices[0].message.content})
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

### Configuring the advisor model (Proxy)

Add the advisor as a named deployment in `model_list` and reference it in `advisor_interception_params`. The proxy router resolves the correct credentials automatically.

```yaml showLineNumbers title="config.yaml"
model_list:
  # Advisor — use a stronger model than your executor (example: o3 as advisor)
  - model_name: my-advisor
    litellm_params:
      model: openai/o3
      api_key: os.environ/OPENAI_API_KEY

  # Or use Anthropic Opus as advisor
  # - model_name: my-advisor
  #   litellm_params:
  #     model: anthropic/claude-opus-4-6
  #     api_key: os.environ/ANTHROPIC_API_KEY

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

### Client request (`litellm_advisor`)

```python showLineNumbers title="Advisor via proxy (OpenAI-compatible client)"
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

### Client request (`advisor_20260301`)

```python showLineNumbers title="Proxy chat completions with native advisor tool type"
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
            "type": "advisor_20260301",
            "name": "advisor",
            "model": "my-advisor",
        }
    ],
    max_tokens=4096,
)
print(response.choices[0].message.content)
```

### Client request (non-Anthropic executor)

Use OpenAI-style `litellm_advisor` when the executor is not Anthropic. The advisor comes from `default_advisor_model`.

```python showLineNumbers title="Proxy with Gemini executor + configured advisor"
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
                    "properties": {"question": {"type": "string"}},
                    "required": ["question"],
                },
            },
        }
    ],
    max_tokens=512,
)
print(response.choices[0].message.content)
```

</TabItem>
</Tabs>

---

## Messages API

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

```python showLineNumbers title="Messages API streaming with advisor tool"
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

#### Non-Anthropic provider (explicit)

```python showLineNumbers title="Advisor Tool with OpenAI executor (custom_llm_provider)"
import asyncio
import litellm

async def main():
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
    print(response["content"][0]["text"])

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
  "role": "assistant",
  "content": [
    {
      "type": "text",
      "text": "Here is the implementation:"
    },
    {
      "type": "server_tool_use",
      "id": "srvtoolu_abc123",
      "name": "advisor"
    },
    {
      "type": "advisor_tool_result",
      "tool_use_id": "srvtoolu_abc123",
      "content": {
        "type": "advisor_result",
        "text": "Use a channel-based coordination pattern..."
      }
    },
    {
      "type": "text",
      "text": "Here's the full implementation..."
    }
  ]
}
```

Pass the full assistant content, including advisor blocks, back on subsequent turns. LiteLLM handles stripping and `provider_specific_fields` where applicable.

### Chat Completions API

For chat completions, the advisor blocks are in `provider_specific_fields` on the response message:

```python title="Accessing advisor results from chat completions"
response = await litellm.acompletion(...)

message = response.choices[0].message
print(message.content)  # final answer

# Advisor trace — available when the advisor was called
psf = message.provider_specific_fields or {}
for block in psf.get("advisor_tool_results", []):
    if block["type"] == "advisor_tool_result":
        print("Advisor said:", block["content"]["text"])
```

```json title="provider_specific_fields structure"
{
  "advisor_tool_results": [
    {
      "type": "server_tool_use",
      "id": "call_abc123",
      "name": "advisor"
    },
    {
      "type": "advisor_tool_result",
      "tool_use_id": "call_abc123",
      "content": {
        "type": "advisor_result",
        "text": "Use a channel-based coordination pattern..."
      }
    }
  ]
}
```

---

## Cost control

Advisor calls run as separate sub-inferences billed at the advisor model's rates. Usage is reported in `usage.iterations[]` (Messages API) or accumulated in `usage` (Chat Completions):

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

Top-level `usage` reflects executor tokens; advisor tokens appear in `iterations` with `type: "advisor_message"`.

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

## Remapping the advisor model

Some clients (e.g. Claude Code) hardcode the advisor tool's `model` field — you cannot change what the client sends. If you still want the advisor sub-call to hit a different model (for cost, availability, or routing reasons), remap the advisor model using `model_group_alias` on the router.

When the advisor tool's `model` resolves through `model_group_alias` to a **non-native Anthropic advisor model**, LiteLLM automatically takes over the orchestration loop — even when the executor is direct Anthropic — and routes the advisor sub-call through the router. The client keeps seeing the original alias in every response surface (`iterations[].model`), so the remap stays opaque to the caller.

```yaml showLineNumbers title="config.yaml — remap claude-opus-4-7 advisor to o3"
model_list:
  - model_name: o3
    litellm_params:
      model: openai/o3
      api_key: os.environ/OPENAI_API_KEY

  - model_name: claude-sonnet
    litellm_params:
      model: anthropic/claude-sonnet-4-6
      api_key: os.environ/ANTHROPIC_API_KEY

router_settings:
  model_group_alias:
    claude-opus-4-7: o3
```

With the config above, a client request that includes:

```json
{
  "type": "advisor_20260301",
  "name": "advisor",
  "model": "claude-opus-4-7"
}
```

will:

1. Run the executor against Anthropic as usual.
2. When the executor calls the advisor, route the sub-call through the router to `openai/o3` using the `o3` deployment's credentials.
3. Emit `iterations[].model == "claude-opus-4-7"` in the response so the client never sees `o3`.

:::info When does the remap trigger?

Only when the resolved model is **not** a native Anthropic advisor (currently `claude-opus-4-6` and `claude-opus-4-7`). If you alias one native advisor model to another (e.g. `claude-opus-4-7 -> claude-opus-4-6`), Anthropic's server-side advisor still handles the request.

:::

### Claude Code quickstart: use any advisor model

If you are using Claude Code and want to run the advisor on a non-Claude model (for example `openai/o3`, Gemini, Bedrock, etc.), use this pattern:

1. Keep Claude Code's advisor tool unchanged:

```json
{
  "type": "advisor_20260301",
  "name": "advisor",
  "model": "claude-opus-4-7"
}
```

2. Map that model name to your actual advisor deployment in LiteLLM:

```yaml showLineNumbers title="config.yaml — Claude Code advisor alias"
model_list:
  - model_name: my-real-advisor
    litellm_params:
      model: openai/o3
      api_key: os.environ/OPENAI_API_KEY

  - model_name: claude-sonnet
    litellm_params:
      model: anthropic/claude-sonnet-4-6
      api_key: os.environ/ANTHROPIC_API_KEY

router_settings:
  model_group_alias:
    claude-opus-4-7: my-real-advisor
```

3. Send requests through `/v1/messages` as usual from Claude Code.

What happens at runtime:

- Claude Code sends `model: "claude-opus-4-7"` in the advisor tool.
- LiteLLM resolves it to `my-real-advisor` and routes the sub-call to `openai/o3`.
- Claude Code still sees `claude-opus-4-7` in response-visible fields (alias stays opaque).

:::tip Troubleshooting (Claude Code + non-native advisor)

- If you see `Invalid value: 'thinking'` from OpenAI, upgrade to a LiteLLM build that includes advisor sub-call message translation for non-Anthropic providers.
- If advisor output appears blank in streamed UI, use a build with `advisor_tool_result` text included in `content_block_start` for fake-streamed advisor responses.
- If spend logs are missing for streamed advisor calls, use a build with deferred logging support for non-`CustomStreamWrapper` anthropic streams.

:::

---

## Additional resources

- [Anthropic Advisor Tool Documentation](https://platform.claude.com/docs/en/agents-and-tools/tool-use/advisor-tool)
- [LiteLLM Tool Calling Guide](https://docs.litellm.ai/docs/completion/function_call)
