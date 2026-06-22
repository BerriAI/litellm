# Server-managed OAuth subscription providers

## Overview

| Property | Details |
|-------|-------|
| Description | Server-side refresh-token adapters for subscription-backed inference providers. Clients call LiteLLM normally and LiteLLM refreshes provider OAuth tokens on the server. |
| Provider Routes on LiteLLM | `claude_max/`, `antigravity2/` |
| Supported Operations | `/chat/completions`, including streaming when the upstream provider supports it |

These providers are intended for private gateways where the LiteLLM server owns the subscription login and clients authenticate only to LiteLLM. Do not expose the server token files to client machines.

## Token storage

Each adapter reads a JSON token file and refreshes it when the access token is near expiry. You can mount files into the default directory or point each provider at an explicit file.

```bash showLineNumbers title="Environment Variables"
export SERVER_OAUTH_TOKEN_DIR="$HOME/.config/litellm/server_oauth"
export CLAUDE_MAX_OAUTH_FILE="$SERVER_OAUTH_TOKEN_DIR/claude-max.json"
```

For secret managers that inject environment variables instead of files, set `CLAUDE_MAX_OAUTH_JSON_B64` to a base64-encoded JSON token payload. LiteLLM restores it to the provider token file with owner-only permissions on first use.

## Claude Max

The `claude_max/` provider uses Anthropic-compatible chat completions with a server-managed Claude OAuth refresh token.

```json showLineNumbers title="claude-max.json"
{
  "access_token": "sk-ant-oat...",
  "refresh_token": "...",
  "expires_at": 1790000000
}
```

```yaml showLineNumbers title="config.yaml"
model_list:
  - model_name: frontier
    litellm_params:
      model: claude_max/claude-opus-4-8
  - model_name: opus-4-8
    litellm_params:
      model: claude_max/claude-opus-4-8
```

Optional overrides:

```bash showLineNumbers title="Claude Max overrides"
export CLAUDE_MAX_TOKEN_URL="https://platform.claude.com/v1/oauth/token"
export CLAUDE_MAX_CLIENT_ID="9d1c250a-e61b-44d9-88ed-5944d1962f5e"
export CLAUDE_MAX_API_BASE="https://api.anthropic.com/v1/messages"
```

## Antigravity 2.0

The `antigravity2/` provider uses Google's official Antigravity 2.0 Python SDK (`google-antigravity`) and its local agent runtime. This is intentionally separate from, and does not use, deprecated Gemini CLI, Cloud Code Assist, or internal gateway contracts.

Install the SDK in the LiteLLM server image because the PyPI wheels include the compiled Antigravity runtime binary:

```bash showLineNumbers title="Server dependency"
pip install google-antigravity
```

Authentication is managed on the LiteLLM server by the official runtime. The Antigravity CLI authenticates through the system keyring and falls back to Google Sign-In when no active session exists; the Python SDK can also use `GEMINI_API_KEY` / `ANTIGRAVITY2_API_KEY` or Vertex settings configured on the server. Clients should not send provider credentials.

```yaml showLineNumbers title="config.yaml"
model_list:
  - model_name: frontier
    litellm_params:
      model: antigravity2/gemini-3.1-pro-preview
  - model_name: antigravity-2-pro
    litellm_params:
      model: antigravity2/gemini-3.1-pro-preview
```

Optional server-side overrides:

```bash showLineNumbers title="Antigravity 2.0 overrides"
export ANTIGRAVITY2_APP_DATA_DIR="/var/lib/litellm/antigravity2"
export ANTIGRAVITY2_API_KEY="$GEMINI_API_KEY"
export ANTIGRAVITY2_VERTEX="false"
export ANTIGRAVITY2_PROJECT="my-gcp-project"
export ANTIGRAVITY2_LOCATION="us-central1"
```

For private inference-gateway usage, LiteLLM disables Antigravity built-in tools and subagents by default, so `antigravity2/` behaves like a chat inference provider rather than an autonomous file-editing/code-execution agent.

## LiteLLM virtual keys

Put these providers behind normal LiteLLM model groups and virtual keys. Client applications should send only the LiteLLM virtual key and the model alias, such as `frontier` or `opus-4-8`; provider OAuth secrets remain server-side.
