# Google Code Assist

Use Google Code Assist models through LiteLLM using Gemini OAuth credentials.

| Property | Details |
|-------|-------|
| Description | Google Code Assist API via Google Cloud Code Assist backend |
| Provider Route on LiteLLM | `google_code_assist/` |
| Supported Endpoints | `/chat/completions` |
| Base Endpoint | `https://cloudcode-pa.googleapis.com` |

## Model Format

Use:

`google_code_assist/<gemini-model>`

Example:

`google_code_assist/gemini-2.5-pro`

## Authentication

Google Code Assist requires OAuth credentials (not API key auth).

LiteLLM checks credentials in this order:

1. `GEMINI_OAUTH_TOKEN`
2. `GEMINI_CREDENTIALS_PATH`
3. `~/.config/litellm/gemini_oauth/oauth_creds.json`
4. `~/.gemini/oauth_creds.json`
5. `~/.config/gcloud/application_default_credentials.json`

### Login via CLI

```bash
export GEMINI_OAUTH_CLIENT_ID="<your-google-oauth-client-id>"
export GEMINI_OAUTH_CLIENT_SECRET="<your-google-oauth-client-secret>"
litellm-proxy gemini login
```

This opens a browser loopback OAuth flow and stores credentials locally.

## Usage - LiteLLM Python SDK

```python
from litellm import completion

response = completion(
    model="google_code_assist/gemini-2.5-pro",
    messages=[{"role": "user", "content": "Write a Python function to parse CSV safely."}],
)

print(response.choices[0].message.content)
```

## Usage - LiteLLM Proxy

```yaml
model_list:
  - model_name: google-code-assist
    litellm_params:
      model: google_code_assist/gemini-2.5-pro
```

```bash
litellm --config /path/to/config.yaml
```

```bash
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <YOUR_LITELLM_KEY>" \
  -d '{
    "model": "google-code-assist",
    "messages": [{"role": "user", "content": "Refactor this code for readability."}]
  }'
```

## Optional Params

- `google_code_assist_project`: Optional project override sent to the Code Assist backend request.

LiteLLM also performs the required `loadCodeAssist` handshake before `generateContent` automatically.
