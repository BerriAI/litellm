import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Custom OAuth2 (client credentials)

## Overview

| Property | Details |
|-------|-------|
| Description | Call any OpenAI-compatible endpoint that is protected by an OAuth2 `client_credentials` gateway. LiteLLM fetches a bearer token from your token endpoint, caches it until it expires, and injects it on every request. |
| Provider Route on LiteLLM | `custom_oauth/` |
| Base URL | Set per model via `api_base` (your OpenAI-compatible endpoint) |
| Supported Operations | [`/chat/completions`](#sample-usage) |

<br />

Use this when your model server speaks the OpenAI chat API but sits behind an
OAuth2 token endpoint (token URL + client id/secret) rather than a static API
key. Point `api_base` at the model server and give LiteLLM the OAuth2 details;
the bearer token is fetched and refreshed automatically.

## Required variables

The OAuth2 fields can be set per model under `litellm_params`, or via environment
variables as a fallback.

| litellm_param | Environment variable | Required | Description |
|-------|-------|-------|-------|
| `oauth_token_url` | `CUSTOM_OAUTH_TOKEN_URL` | yes | OAuth2 token endpoint |
| `oauth_client_id` | `CUSTOM_OAUTH_CLIENT_ID` | yes | Client id |
| `oauth_client_secret` | `CUSTOM_OAUTH_CLIENT_SECRET` | yes | Client secret |
| `oauth_scope` | `CUSTOM_OAUTH_SCOPE` | no | Space-delimited scope(s) |
| `api_base` | - | yes | Your OpenAI-compatible endpoint |

## Usage - LiteLLM Python SDK

```python showLineNumbers title="custom_oauth completion"
import os
import litellm
from litellm import completion

response = completion(
    model="custom_oauth/my-model",
    messages=[{"role": "user", "content": "Hello, how are you?"}],
    api_base="https://gateway.internal/v1",
    oauth_token_url="https://idp.internal/oauth/token",
    oauth_client_id=os.environ["MY_CLIENT_ID"],
    oauth_client_secret=os.environ["MY_CLIENT_SECRET"],
    oauth_scope="my-scope",  # optional
)

print(response)
```

## Usage - LiteLLM Proxy

```yaml showLineNumbers title="config.yaml"
model_list:
  - model_name: my-secure-llm
    litellm_params:
      model: custom_oauth/my-model
      api_base: https://gateway.internal/v1
      oauth_token_url: https://idp.internal/oauth/token
      oauth_client_id: os.environ/MY_CLIENT_ID
      oauth_client_secret: os.environ/MY_CLIENT_SECRET
      oauth_scope: my-scope
```

```bash showLineNumbers title="Start the proxy"
litellm --config config.yaml
```

```bash showLineNumbers title="Sample request"
curl http://0.0.0.0:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{
    "model": "my-secure-llm",
    "messages": [{"role": "user", "content": "Hello, how are you?"}]
  }'
```

## How it works

LiteLLM performs the OAuth2 `client_credentials` grant against `oauth_token_url`,
reads `access_token` and `expires_in` from the response, and caches the token in
process until shortly before it expires (refreshing automatically). The token is
sent as `Authorization: Bearer <token>` to `api_base`, which is otherwise treated
as a standard OpenAI-compatible endpoint, so request/response shaping and
streaming behave the same as any OpenAI-compatible provider.

This is the 2-legged (machine-to-machine) flow only. Interactive / authorization-code
login, per-user tokens, and refresh-token grants are out of scope.
