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
| Supported Endpoints | `/chat/completions` |
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
    messages=[{"role": "user", "content": "Write a Python function to calculate fibonacci numbers"}],
    extra_headers={
        "editor-version": "vscode/1.85.1",
        "Copilot-Integration-Id": "vscode-chat"
    }
)
print(response)
```

```python showLineNumbers title="GitHub Copilot Chat Completion - Streaming"
from litellm import completion

stream = completion(
    model="github_copilot/gpt-4",
    messages=[{"role": "user", "content": "Explain async/await in Python"}],
    stream=True,
    extra_headers={
        "editor-version": "vscode/1.85.1",
        "Copilot-Integration-Id": "vscode-chat"
    }
)

for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
```

## Usage - LiteLLM Proxy

Add the following to your LiteLLM Proxy configuration file:

```yaml showLineNumbers title="config.yaml"
model_list:
  - model_name: github_copilot/gpt-4
    litellm_params:
      model: github_copilot/gpt-4
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
    messages=[{"role": "user", "content": "How do I optimize this SQL query?"}],
    extra_headers={
        "editor-version": "vscode/1.85.1",
        "Copilot-Integration-Id": "vscode-chat"
    }
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
    api_key="your-proxy-api-key",
    extra_headers={
        "editor-version": "vscode/1.85.1",
        "Copilot-Integration-Id": "vscode-chat"
    }
)

print(response.choices[0].message.content)
```

</TabItem>

<TabItem value="curl" label="cURL">

```bash showLineNumbers title="GitHub Copilot via Proxy - cURL"
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-proxy-api-key" \
  -H "editor-version: vscode/1.85.1" \
  -H "Copilot-Integration-Id: vscode-chat" \
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
```

### Headers

GitHub Copilot supports various editor-specific headers:

```python showLineNumbers title="Common Headers"
extra_headers = {
    "editor-version": "vscode/1.85.1",           # Editor version
    "editor-plugin-version": "copilot/1.155.0",  # Plugin version
    "Copilot-Integration-Id": "vscode-chat",     # Integration ID
    "user-agent": "GithubCopilot/1.155.0"       # User agent
}
```

