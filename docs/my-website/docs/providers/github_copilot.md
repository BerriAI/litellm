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

GitHub Copilot uses OAuth Device Code flow for authentication. LiteLLM supports two authentication modes:

- **LiteLLM Proxy**: Use named credentials created via the credential API or UI. The interactive device flow is handled by the proxy on your behalf — credentials are stored and reused automatically.
- **Python SDK**: Credentials are read from `~/.config/litellm/github_copilot/access-token` on disk (file-based, for backward compatibility).

:::info

If you hit a GitHub Copilot model without a configured credential (proxy) or access-token file (SDK), LiteLLM returns an `AuthenticationError` pointing to this page.

:::

## Usage - LiteLLM Python SDK

### Setup (first time)

Authenticate once by running the login command. This stores your access token in `~/.config/litellm/github_copilot/access-token` for future use.

```bash showLineNumbers title="Authenticate via device flow (SDK only)"
litellm --login github_copilot
```

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

The proxy requires a named credential. See [Credential-Based Authentication](#credential-based-authentication-proxy) below.

Add the following to your LiteLLM Proxy configuration file:

```yaml showLineNumbers title="config.yaml"
model_list:
  - model_name: github_copilot/gpt-4
    litellm_params:
      model: github_copilot/gpt-4
      litellm_credential_name: my-copilot   # named credential (required for proxy)
  - model_name: github_copilot/gpt-5.1-codex
    model_info:
      mode: responses
    litellm_params:
      model: github_copilot/gpt-5.1-codex
      litellm_credential_name: my-copilot
  - model_name: github_copilot/text-embedding-ada-002
    model_info:
      mode: embedding
    litellm_params:
      model: github_copilot/text-embedding-ada-002
      litellm_credential_name: my-copilot
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

## Credential-Based Authentication (Proxy)

The LiteLLM Proxy uses a stateless OAuth Device Code flow. Nothing is stored in the database until the GitHub token is successfully obtained — the client holds the `device_code` between API calls.

You can complete the flow via the **LiteLLM UI** (Models → Credentials → Add Credential → GitHub Copilot) or with the curl steps below.

### Step 1: Initiate the Device Code Flow

```bash showLineNumbers title="Start GitHub OAuth"
curl -X POST http://localhost:4000/credentials/github_copilot/initiate \
  -H "Authorization: Bearer your-proxy-api-key"
```

Response:

```json
{
  "device_code": "xxx",
  "user_code": "ABCD-1234",
  "verification_uri": "https://github.com/login/device",
  "poll_interval_ms": 5000,
  "expires_in": 900
}
```

### Step 2: Authorize on GitHub

Visit the `verification_uri` and enter the `user_code` to authorize LiteLLM.

### Step 3: Poll for Completion

Poll the status endpoint until the flow completes. Use `device_code` from step 1.
Use `poll_interval_ms` from the initiate response as your default polling interval.

```bash showLineNumbers title="Check authorization status"
curl -X POST http://localhost:4000/credentials/github_copilot/status \
  -H "Authorization: Bearer your-proxy-api-key" \
  -H "Content-Type: application/json" \
  -d '{"device_code": "xxx"}'
```

Possible responses:

```json title="Still waiting for user"
{"status": "pending"}
```

```json title="GitHub is being polled too fast — wait before retrying"
{"status": "pending", "retry_after_ms": 10000}
```

If `retry_after_ms` is present, you **must** wait that many milliseconds before calling `/status` again. Ignoring it causes GitHub to keep increasing the required interval.

```json title="User authorized successfully"
{"status": "complete", "access_token": "ghu_xxx"}
```

```json title="Flow expired or denied"
{"status": "failed", "error": "The device code has expired."}
```

### Step 4: Store as a Named Credential

Once you have the `access_token`, store it as a named credential:

```bash showLineNumbers title="Save credential"
curl -X POST http://localhost:4000/credentials \
  -H "Authorization: Bearer your-proxy-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "credential_name": "my-copilot",
    "credential_values": {"api_key": "ghu_xxx"},
    "credential_info": {"custom_llm_provider": "github_copilot"}
  }'
```

### Step 5: Attach Credential to a Model

Reference the credential in your config (or via the UI under Models → Add Model → Existing Credentials):

```yaml showLineNumbers title="config.yaml"
model_list:
  - model_name: copilot-gpt4
    litellm_params:
      model: github_copilot/gpt-4
      litellm_credential_name: my-copilot
```

### Multiple Accounts

Create multiple credentials with different names to support different GitHub accounts:

```yaml showLineNumbers title="config.yaml - Multiple accounts"
model_list:
  - model_name: copilot-team-a
    litellm_params:
      model: github_copilot/gpt-4
      litellm_credential_name: team-a-copilot
  - model_name: copilot-team-b
    litellm_params:
      model: github_copilot/gpt-4
      litellm_credential_name: team-b-copilot
```

## Configuration

### Environment Variables

You can customize token storage locations (SDK / file-based mode):

```bash showLineNumbers title="Environment Variables"
# Optional: Custom token directory
export GITHUB_COPILOT_TOKEN_DIR="~/.config/litellm/github_copilot"

# Optional: Custom access token file name
export GITHUB_COPILOT_ACCESS_TOKEN_FILE="access-token"

# Optional: Custom API key file name
export GITHUB_COPILOT_API_KEY_FILE="api-key.json"
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
