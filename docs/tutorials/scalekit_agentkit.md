# Scalekit with LiteLLM

Add authenticated tool calls to your LiteLLM-powered agents. [Scalekit](https://docs.scalekit.com/agentkit/overview/) manages OAuth flows, token storage, and API execution for 100+ third-party apps (Gmail, GitHub, Slack, Salesforce, etc.) — your agent picks tools at runtime and LiteLLM routes the model calls to any provider.

## Overview

- Fetch user-scoped tool definitions from Scalekit and pass them as function schemas to `litellm.completion()`
- Switch models freely — the same tool definitions work across OpenAI, Anthropic, Bedrock, Vertex AI, and every other provider LiteLLM supports
- Execute tool calls through Scalekit — no API keys, endpoints, or auth headers to manage per third-party app

## Prerequisites

- Python 3.9+
- A [Scalekit account](https://app.scalekit.com) with a connection configured (this tutorial uses Gmail)
- API keys for at least one LLM provider, or a running LiteLLM proxy
- Scalekit API credentials (`SCALEKIT_CLIENT_ID`, `SCALEKIT_CLIENT_SECRET`, `SCALEKIT_ENV_URL`) from Dashboard → **Developers** → **API Credentials**

## 1. Install Dependencies

```bash
pip install litellm scalekit-sdk-python
```

## 2. Initialize Clients

```python showLineNumbers title="setup.py"
import os
import json
import litellm
import scalekit.client
from google.protobuf.json_format import MessageToDict  # installed with scalekit-sdk-python

scalekit_client = scalekit.client.ScalekitClient(
    client_id=os.getenv("SCALEKIT_CLIENT_ID"),
    client_secret=os.getenv("SCALEKIT_CLIENT_SECRET"),
    env_url=os.getenv("SCALEKIT_ENV_URL"),
)
actions = scalekit_client.actions
```

## 3. Authorize a User

Create a connected account and complete the OAuth flow. Once the account status is `ACTIVE`, Scalekit can execute tools on behalf of the user.

```python showLineNumbers title="authorize.py"
connection_name = os.getenv("GMAIL_CONNECTION_NAME", "gmail")

response = actions.get_or_create_connected_account(
    connection_name=connection_name,
    identifier="user_123",  # your app's user ID
)
connected_account = response.connected_account

if connected_account.status != "ACTIVE":
    link = actions.get_authorization_link(
        connection_name=connection_name,
        identifier="user_123",
    )
    print("Authorize Gmail:", link.link)
    input("Press Enter after completing authorization...")
```

## 4. Fetch Scoped Tools

`list_scoped_tools` returns only the tools this specific user is authorized to call. Convert them to OpenAI's function-calling format — the same format LiteLLM normalizes to across all providers.

```python showLineNumbers title="fetch_tools.py"
scoped_response, _ = actions.tools.list_scoped_tools(
    identifier="user_123",
    filter={"connection_names": [connection_name]},
    page_size=100,
)

# Convert to OpenAI function-calling format (used by litellm for all providers)
llm_tools = [
    {
        "type": "function",
        "function": {
            "name": MessageToDict(t.tool).get("definition", {}).get("name"),
            "description": MessageToDict(t.tool).get("definition", {}).get("description", ""),
            "parameters": MessageToDict(t.tool).get("definition", {}).get("input_schema", {}),
        },
    }
    for t in scoped_response.tools
]
```

## 5. Run the Agent Loop

Call `litellm.completion()` with the tool definitions. When the model returns tool calls, execute them through Scalekit and feed the results back. Change the `model` parameter to switch providers — no other code changes needed.

```python showLineNumbers title="agent_loop.py"
messages = [{"role": "user", "content": "Fetch my last 5 unread emails and summarize them"}]

while True:
    response = litellm.completion(
        model="anthropic/claude-sonnet-4-20250514",  # swap to any litellm-supported model
        tools=llm_tools,
        messages=messages,
    )
    message = response.choices[0].message

    if not message.tool_calls:
        print(message.content)
        break

    # Append assistant message with tool calls
    messages.append(message)

    # Execute each tool call through Scalekit
    for tc in message.tool_calls:
        result = actions.execute_tool(
            tool_name=tc.function.name,
            identifier="user_123",
            tool_input=json.loads(tc.function.arguments),
        )
        messages.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": str(result.data),
        })
```

## 6. Complete Working Example

Full end-to-end script — copy and run:

```python showLineNumbers title="scalekit_agent.py"
import os
import json
import litellm
import scalekit.client
from google.protobuf.json_format import MessageToDict

# --- Configuration ---
MODEL = os.getenv("MODEL", "anthropic/claude-sonnet-4-20250514")
CONNECTION_NAME = os.getenv("GMAIL_CONNECTION_NAME", "gmail")
USER_ID = "user_123"

# --- Initialize ---
scalekit_client = scalekit.client.ScalekitClient(
    client_id=os.getenv("SCALEKIT_CLIENT_ID"),
    client_secret=os.getenv("SCALEKIT_CLIENT_SECRET"),
    env_url=os.getenv("SCALEKIT_ENV_URL"),
)
actions = scalekit_client.actions

# --- Authorize user ---
response = actions.get_or_create_connected_account(
    connection_name=CONNECTION_NAME,
    identifier=USER_ID,
)
if response.connected_account.status != "ACTIVE":
    link = actions.get_authorization_link(
        connection_name=CONNECTION_NAME,
        identifier=USER_ID,
    )
    print("Authorize Gmail:", link.link)
    input("Press Enter after completing authorization...")

# --- Fetch tools ---
scoped_response, _ = actions.tools.list_scoped_tools(
    identifier=USER_ID,
    filter={"connection_names": [CONNECTION_NAME]},
    page_size=100,
)
llm_tools = [
    {
        "type": "function",
        "function": {
            "name": MessageToDict(t.tool).get("definition", {}).get("name"),
            "description": MessageToDict(t.tool).get("definition", {}).get("description", ""),
            "parameters": MessageToDict(t.tool).get("definition", {}).get("input_schema", {}),
        },
    }
    for t in scoped_response.tools
]
print(f"Loaded {len(llm_tools)} tools for {CONNECTION_NAME}")

# --- Agent loop ---
messages = [{"role": "user", "content": "Fetch my last 5 unread emails and summarize them"}]

while True:
    response = litellm.completion(model=MODEL, tools=llm_tools, messages=messages)
    message = response.choices[0].message

    if not message.tool_calls:
        print(message.content)
        break

    messages.append(message)
    for tc in message.tool_calls:
        print(f"  Calling tool: {tc.function.name}")
        result = actions.execute_tool(
            tool_name=tc.function.name,
            identifier=USER_ID,
            tool_input=json.loads(tc.function.arguments),
        )
        messages.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": str(result.data),
        })
```

Switch models by changing the `MODEL` environment variable:

```bash
# OpenAI
MODEL=gpt-4o python scalekit_agent.py

# Anthropic
MODEL=anthropic/claude-sonnet-4-20250514 python scalekit_agent.py

# AWS Bedrock
MODEL=bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0 python scalekit_agent.py

# Via LiteLLM Proxy
OPENAI_API_BASE=http://localhost:4000 OPENAI_API_KEY=sk-1234 MODEL=claude-sonnet-4 python scalekit_agent.py
```

## Route Through LiteLLM Proxy for Cost Tracking and Rate Limits

If you're running a LiteLLM proxy, point your agent at it for centralized model management, cost tracking, and rate limiting. The agent code stays the same — set the proxy URL:

```python showLineNumbers title="proxy_agent.py"
import litellm

# Point litellm at your proxy
response = litellm.completion(
    model="claude-sonnet-4",                          # model name from your proxy config
    api_base="http://localhost:4000",                  # proxy URL
    api_key="sk-1234",                                 # proxy virtual key
    tools=llm_tools,
    messages=messages,
)
```

Or use environment variables so no code changes are needed:

```bash
export OPENAI_API_BASE="http://localhost:4000"
export OPENAI_API_KEY="sk-1234"
python scalekit_agent.py
```

## End-to-End Example: Inbox Triage Agent

For a production-style example that combines Scalekit tool execution with per-stage model routing through LiteLLM, see [litellm-agentkit-inbox-triage](https://github.com/scalekit-developers/litellm-agentkit-inbox-triage). It demonstrates:

- Polling Gmail and classifying threads with different models per pipeline stage
- Routing to GitHub repos using keyword rules and LLM tie-breaking
- Searching related GitHub issues through a Scalekit tool-calling loop
- Notifying Slack and waiting for human approval before creating issues or sending replies

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `execute_tool` returns "connection not found" | The `connection_name` must match the exact label in Dashboard → **AgentKit** → **Connections** (including case). Use an env var instead of hardcoding. |
| Connected account stays in `PENDING` | The user hasn't completed the OAuth flow. Regenerate the authorization link and have them open it in a browser. |
| Model returns text instead of tool calls | Not all models support function calling. Use a model that does (GPT-4o, Claude Sonnet/Opus, Gemini Pro). Check [supported providers](../providers/). |
| `litellm.completion()` raises an auth error | Verify your LLM provider API key is set (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.) or that your proxy URL and key are correct. |

## Related Resources

- [Scalekit Docs](https://docs.scalekit.com/agentkit/overview/) — Full documentation
- [Built-in Tools Reference](https://docs.scalekit.com/agentkit/tools/scalekit-optimized-tools/) — Tool calling across 100+ connectors
- [Supported Connectors](https://docs.scalekit.com/agentkit/connectors/) — Gmail, GitHub, Slack, Salesforce, and more
- [LiteLLM Proxy Quick Start](../proxy/quick_start) — Set up centralized model routing
- [LiteLLM Function Calling](../completion/function_call) — Function calling docs