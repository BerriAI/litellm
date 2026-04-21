import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# GitHub Copilot

https://docs.github.com/en/copilot

:::tip

**We support GitHub Copilot Chat API with automatic authentication handling**

:::

| Property | Details |
|-------|-------|
| Description | GitHub Copilot Chat API provides access to GitHub's AI-powered coding assistant. |
| Provider Route on LiteLLM | `github_copilot/` |
| Supported Endpoints | `/chat/completions`, `/embeddings` |
| API Reference | [GitHub Copilot docs](https://docs.github.com/en/copilot) |

## Authentication

GitHub Copilot uses OAuth device flow for authentication. On first use, you'll be prompted to authenticate via GitHub:

1. LiteLLM will display a device code and verification URL
2. Visit the URL and enter the code to authenticate
3. Your credentials will be stored locally for future use

## Usage - LiteLLM Python SDK

### Chat Completion

```python showLineNumbers title="GitHub Copilot Chat Completion"
from litellm import completion

response = completion(
    model="github_copilot/gpt-4",
    messages=[
        {"role": "system", "content": "You are a helpful coding assistant"},
        {"role": "user", "content": "Write a Python function to calculate fibonacci numbers"}
    ]
)
print(response)
```

```python showLineNumbers title="GitHub Copilot Chat Completion - Streaming"
from litellm import completion

stream = completion(
    model="github_copilot/gpt-4",
    messages=[{"role": "user", "content": "Explain async/await in Python"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
```

### Responses

For GPT Codex models, only responses API is supported.

```python showLineNumbers title="GitHub Copilot Responses"
import litellm

response = await litellm.aresponses(
    model="github_copilot/gpt-5.1-codex",
    input="Write a Python hello world",
    max_output_tokens=500
)

print(response)
```

### Embedding

```python showLineNumbers title="GitHub Copilot Embedding"
import litellm

response = litellm.embedding(
    model="github_copilot/text-embedding-3-small",
    input=["good morning from litellm"]
)
print(response)
```

## Usage - LiteLLM Proxy

Add the following to your LiteLLM Proxy configuration file:

```yaml showLineNumbers title="config.yaml"
model_list:
  - model_name: github_copilot/gpt-4
    litellm_params:
      model: github_copilot/gpt-4
  - model_name: github_copilot/gpt-5.1-codex
    model_info:
      mode: responses
    litellm_params:
      model: github_copilot/gpt-5.1-codex
  - model_name: github_copilot/text-embedding-ada-002
    model_info:
      mode: embedding
    litellm_params:
      model: github_copilot/text-embedding-ada-002
```

Start your LiteLLM Proxy server:

```bash showLineNumbers title="Start LiteLLM Proxy"
litellm --config config.yaml

# RUNNING on http://0.0.0.0:4000
```

<Tabs>
<TabItem value="openai-sdk" label="OpenAI SDK">

```python showLineNumbers title="GitHub Copilot via Proxy - Non-streaming"
from openai import OpenAI

# Initialize client with your proxy URL
client = OpenAI(
    base_url="http://localhost:4000",  # Your proxy URL
    api_key="your-proxy-api-key"       # Your proxy API key
)

# Non-streaming response
response = client.chat.completions.create(
    model="github_copilot/gpt-4",
    messages=[{"role": "user", "content": "How do I optimize this SQL query?"}]
)

print(response.choices[0].message.content)
```

</TabItem>

<TabItem value="litellm-sdk" label="LiteLLM SDK">

```python showLineNumbers title="GitHub Copilot via Proxy - LiteLLM SDK"
import litellm

# Configure LiteLLM to use your proxy
response = litellm.completion(
    model="litellm_proxy/github_copilot/gpt-4",
    messages=[{"role": "user", "content": "Review this code for bugs"}],
    api_base="http://localhost:4000",
    api_key="your-proxy-api-key"
)

print(response.choices[0].message.content)
```

</TabItem>

<TabItem value="curl" label="cURL">

```bash showLineNumbers title="GitHub Copilot via Proxy - cURL"
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-proxy-api-key" \
  -d '{
    "model": "github_copilot/gpt-4",
    "messages": [{"role": "user", "content": "Explain this error message"}]
  }'
```

</TabItem>
</Tabs>

## Getting Started

1. Ensure you have GitHub Copilot access (paid GitHub subscription required)
2. Run your first LiteLLM request - you'll be prompted to authenticate
3. Follow the device flow authentication process
4. Start making requests to GitHub Copilot through LiteLLM

## Configuration

### Environment Variables

You can customize token storage locations:

```bash showLineNumbers title="Environment Variables"
# Optional: Custom token directory
export GITHUB_COPILOT_TOKEN_DIR="~/.config/litellm/github_copilot"

# Optional: Custom access token file name
export GITHUB_COPILOT_ACCESS_TOKEN_FILE="access-token"

# Optional: Custom API key file name
export GITHUB_COPILOT_API_KEY_FILE="api-key.json"

# Optional: Custom Copilot endpoints for authentication and usage
# (needed when using GitHub Enterprise subscriptions with custom endpoints or self-hosted GitHub servers
export GITHUB_COPILOT_API_BASE="https://copilot-api.my-company.ghe.com"
export GITHUB_COPILOT_DEVICE_CODE_URL="https://my-company.ghe.com/login/device/code"
export GITHUB_COPILOT_ACCESS_TOKEN_URL="https://my-company.ghe.com/login/oauth/access_token"
export GITHUB_COPILOT_API_KEY_URL="https://my-company.ghe.com/api/v3/copilot_internal/v2/token"
```

### Headers

LiteLLM automatically injects the required GitHub Copilot headers (simulating VSCode). You don't need to specify them manually.

If you want to override the defaults (e.g., to simulate a different editor), you can use `extra_headers`:

```python showLineNumbers title="Custom Headers (Optional)"
extra_headers = {
    "editor-version": "vscode/1.85.1",           # Editor version
    "editor-plugin-version": "copilot/1.155.0",  # Plugin version
    "Copilot-Integration-Id": "vscode-chat",     # Integration ID
    "user-agent": "GithubCopilot/1.155.0"        # User agent
}
```

## Premium Request Billing

GitHub Copilot charges a **premium request** for each conversation that starts with only user or system messages. LiteLLM's adapter handles billing automatically by setting the `X-Initiator` header on every outbound request.

### How Billing Works

| Turn | Messages Sent | X-Initiator | Charged? |
|------|---------------|-------------|----------|
| 1 | `[{"role": "user", ...}]` | `user` | Yes — 1 premium request |
| 2+ | `[..., {"role": "assistant", ...}, {"role": "user", ...}]` | `agent` | No — continuation |

**Rule:** Once the messages list contains an `assistant` (or `tool`) role, LiteLLM sets `X-Initiator: agent` on all subsequent turns. GitHub Copilot treats those turns as free continuations of the same conversation.

This applies equally to both the `/chat/completions` endpoint (e.g., `claude-sonnet-4.5`, `gpt-4o`) and the Responses API endpoint (e.g., `gpt-5.1-codex`).

:::note

The `disable_copilot_system_to_assistant` setting does **not** affect premium request billing. `X-Initiator` is determined from the original message roles before any system-to-assistant transformation occurs.

:::

### Recommended Configuration

For multi-turn agentic workflows (Claude Code, Codex CLI, agent plans), the following proxy configuration is recommended:

```yaml showLineNumbers title="config.yaml — recommended for agentic workflows"
model_list:
  - model_name: "claude-sonnet-4.5"
    litellm_params:
      model: "github_copilot/claude-sonnet-4.5"
      api_key: "os.environ/GITHUB_COPILOT_API_KEY"

  - model_name: "gpt-5.1-codex"
    litellm_params:
      model: "github_copilot/gpt-5.1-codex"
      api_key: "os.environ/GITHUB_COPILOT_API_KEY"

litellm_settings:
  drop_params: true                           # Recommended: drops unsupported params silently
  disable_copilot_system_to_assistant: false  # Default: converts system→assistant messages
```

No configuration changes are required from existing LiteLLM GitHub Copilot users — the default settings produce correct billing behavior.

**If you use native system messages** (GitHub Copilot now supports the `system` role natively):

```yaml showLineNumbers title="config.yaml — disable system→assistant conversion"
litellm_settings:
  disable_copilot_system_to_assistant: true  # Keep system role as-is
```

### Troubleshooting: Verifying Premium Request Consumption

Use this 5-step procedure to confirm LiteLLM is billing correctly for multi-turn conversations:

**Step 1 — Record "Before" count**

Open https://github.com/settings/copilot → **Usage and billing** tab. Note the current premium request count and timestamp.

**Step 2 — Start proxy with debug logging**

```bash showLineNumbers title="Start proxy with debug logging"
GITHUB_COPILOT_API_KEY=$GITHUB_COPILOT_API_KEY LITELLM_LOG=DEBUG \
  litellm --config config.yaml --port 4000
```

**Step 3 — Send a 2-turn test conversation**

Turn 1 (charges 1 premium — user-only messages):

```bash showLineNumbers title="Turn 1 — should charge 1 premium request"
curl -s -X POST http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer test-key" \
  -d '{
    "model": "claude-sonnet-4.5",
    "messages": [{"role": "user", "content": "Write a Python Fibonacci function."}],
    "max_tokens": 300
  }' | python3 -m json.tool
```

Turn 2 (no charge — includes assistant history). Replace `<REPLY>` with the `content` from Turn 1:

```bash showLineNumbers title="Turn 2 — should NOT charge (agent continuation)"
curl -s -X POST http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer test-key" \
  -d '{
    "model": "claude-sonnet-4.5",
    "messages": [
      {"role": "user", "content": "Write a Python Fibonacci function."},
      {"role": "assistant", "content": "<REPLY>"},
      {"role": "user", "content": "Add memoization."}
    ],
    "max_tokens": 300
  }' | python3 -m json.tool
```

**Step 4 — Check proxy logs**

In the proxy terminal, look for these log lines:

```
[<id>] X-Initiator determined: user (message_count=1)        ← Turn 1 (PREMIUM)
[<id>] X-Initiator determined: agent (message_count=3, ...)  ← Turn 2 (FREE)
```

If Turn 2 also shows `user`, the fix is not applied — check your LiteLLM version.

**Step 5 — Record "After" count**

Wait ~30 seconds, then refresh https://github.com/settings/copilot → **Usage and billing**.

```
Delta = After − Before
Expected: 1 (Turn 1 only)
```

If the delta is greater than 1 for this 2-turn test, open an issue at https://github.com/BerriAI/litellm with the proxy log output.

