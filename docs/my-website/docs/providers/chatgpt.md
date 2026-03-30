import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# ChatGPT Subscription

Use ChatGPT Pro/Max subscription models through LiteLLM with OAuth device flow authentication.

| Property | Details |
|-------|-------|
| Description | ChatGPT subscription access (Codex + GPT-5.3/5.4 family) via ChatGPT backend API |
| Provider Route on LiteLLM | `chatgpt/` |
| Supported Endpoints | `/responses`, `/chat/completions` (bridged to Responses for supported models) |
| API Reference | https://chatgpt.com |

ChatGPT subscription access is native to the Responses API. Chat Completions requests are bridged to Responses for supported models (for example `chatgpt/gpt-5.4`).

Notes:
- The ChatGPT subscription backend rejects token limit fields (`max_tokens`, `max_output_tokens`, `max_completion_tokens`) and `metadata`. LiteLLM strips these fields for this provider.
- `/v1/chat/completions` honors `stream`. When `stream` is false (default), LiteLLM aggregates the Responses stream into a single JSON response.

## Authentication

ChatGPT subscription access uses an OAuth Device Code flow. LiteLLM supports two authentication modes:

- **LiteLLM Proxy**: Use named credentials created via the credential API or UI. The interactive device flow is handled by the proxy on your behalf — credentials are stored and reused automatically.
- **Python SDK**: Credentials are stored locally in `~/.config/litellm/chatgpt/auth.json` on disk (file-based, for backward compatibility).

:::info

If you hit a ChatGPT model without a configured credential (proxy) or auth file (SDK), LiteLLM returns an `AuthenticationError`.

:::

## Usage - LiteLLM Python SDK

### Setup (first time)

Authenticate once by running the login command. This stores your refresh token in `~/.config/litellm/chatgpt/auth.json` for future use.

```bash showLineNumbers title="Authenticate via device flow (SDK only)"
litellm --login chatgpt
```

### Responses (recommended for Codex models)

```python showLineNumbers title="ChatGPT Responses"
import litellm

response = litellm.responses(
    model="chatgpt/gpt-5.3-codex",
    input="Write a Python hello world"
)

print(response)
```

### Chat Completions (bridged to Responses)

```python showLineNumbers title="ChatGPT Chat Completions"
import litellm

response = litellm.completion(
    model="chatgpt/gpt-5.4",
    messages=[{"role": "user", "content": "Write a Python hello world"}]
)

print(response)
```

## Usage - LiteLLM Proxy

The proxy requires a named credential. See [Credential-Based Authentication](#credential-based-authentication-proxy) below.

```yaml showLineNumbers title="config.yaml"
model_list:
  - model_name: chatgpt/gpt-5.4
    model_info:
      mode: responses
    litellm_params:
      model: chatgpt/gpt-5.4
      litellm_credential_name: my-chatgpt   # named credential (required for proxy)
  - model_name: chatgpt/gpt-5.4-pro
    model_info:
      mode: responses
    litellm_params:
      model: chatgpt/gpt-5.4-pro
      litellm_credential_name: my-chatgpt
  - model_name: chatgpt/gpt-5.3-codex
    model_info:
      mode: responses
    litellm_params:
      model: chatgpt/gpt-5.3-codex
      litellm_credential_name: my-chatgpt
```

```bash showLineNumbers title="Start LiteLLM Proxy"
litellm --config config.yaml
```

<Tabs>
<TabItem value="openai-sdk" label="OpenAI SDK">

```python showLineNumbers title="ChatGPT via Proxy - OpenAI SDK"
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:4000",
    api_key="your-proxy-api-key"
)

response = client.chat.completions.create(
    model="chatgpt/gpt-5.4",
    messages=[{"role": "user", "content": "Write a Python hello world"}]
)

print(response.choices[0].message.content)
```

</TabItem>

<TabItem value="curl" label="cURL">

```bash showLineNumbers title="ChatGPT via Proxy - cURL"
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-proxy-api-key" \
  -d '{
    "model": "chatgpt/gpt-5.4",
    "messages": [{"role": "user", "content": "Write a Python hello world"}]
  }'
```

</TabItem>
</Tabs>

## Credential-Based Authentication (Proxy)

The LiteLLM Proxy uses a stateless OAuth Device Code flow. Nothing is stored in the database until the tokens are successfully obtained — the client holds the `device_auth_id` and `user_code` between API calls.

You can complete the flow via the **LiteLLM UI** (LLM Credentials → Add Credential → ChatGPT) or with the curl steps below.

### Step 1: Initiate the Device Code Flow

```bash showLineNumbers title="Start ChatGPT OAuth"
curl -X POST http://localhost:4000/credentials/chatgpt/initiate \
  -H "Authorization: Bearer your-proxy-api-key"
```

Response:

```json
{
  "device_auth_id": "xxx",
  "user_code": "ABCD-1234",
  "verification_uri": "https://auth0.openai.com/activate",
  "poll_interval_ms": 5000
}
```

### Step 2: Authorize on OpenAI

Visit the `verification_uri` and enter the `user_code` to authorize LiteLLM.

### Step 3: Poll for Completion

Poll the status endpoint until the flow completes. Use `device_auth_id` and `user_code` from step 1.

```bash showLineNumbers title="Check authorization status"
curl -X POST http://localhost:4000/credentials/chatgpt/status \
  -H "Authorization: Bearer your-proxy-api-key" \
  -H "Content-Type: application/json" \
  -d '{"device_auth_id": "xxx", "user_code": "ABCD-1234"}'
```

Possible responses:

```json title="Still waiting for user"
{"status": "pending"}
```

```json title="User authorized successfully"
{"status": "complete", "refresh_token": "xxx", "account_id": "user-xxx"}
```

```json title="Flow expired or denied"
{"status": "failed", "error": "Token exchange failed: HTTP 400"}
```

### Step 4: Store as a Named Credential

Once you have the `refresh_token`, store it as a named credential:

```bash showLineNumbers title="Save credential"
curl -X POST http://localhost:4000/credentials \
  -H "Authorization: Bearer your-proxy-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "credential_name": "my-chatgpt",
    "credential_values": {"api_key": "xxx"},
    "credential_info": {"custom_llm_provider": "chatgpt"}
  }'
```

:::note
The `api_key` field stores the OpenAI refresh token. LiteLLM automatically exchanges it for short-lived access tokens on each request.
:::

### Step 5: Attach Credential to a Model

Reference the credential in your config (or via the UI under Models → Add Model → Existing Credentials):

```yaml showLineNumbers title="config.yaml"
model_list:
  - model_name: chatgpt-codex
    model_info:
      mode: responses
    litellm_params:
      model: chatgpt/gpt-5.3-codex
      litellm_credential_name: my-chatgpt
```

### Re-authorization

If a credential stops working (e.g. the refresh token is revoked), use the **Re-authorize** button (key icon) on the LLM Credentials page to redo the device code flow and update the stored token.

## Configuration

### Environment Variables

- `CHATGPT_TOKEN_DIR`: Custom token storage directory
- `CHATGPT_AUTH_FILE`: Auth file name (default: `auth.json`)
- `CHATGPT_API_BASE`: Override API base (default: `https://chatgpt.com/backend-api/codex`)
- `OPENAI_CHATGPT_API_BASE`: Alias for `CHATGPT_API_BASE`
- `CHATGPT_ORIGINATOR`: Override the `originator` header value
- `CHATGPT_USER_AGENT`: Override the `User-Agent` header value
- `CHATGPT_USER_AGENT_SUFFIX`: Optional suffix appended to the `User-Agent` header
