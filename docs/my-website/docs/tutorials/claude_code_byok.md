# Claude Code with Bring Your Own Key (BYOK)

Use Claude Code with your own Anthropic API key through the LiteLLM proxy. When you use Claude's `/login` with your Anthropic account, your API key is sent as `x-api-key`. With BYOK enabled, LiteLLM forwards your key to Anthropic instead of using proxy-configured keys — so you pay Anthropic directly while still benefiting from LiteLLM's routing, logging, and guardrails.

## How It Works

1. **Claude Code `/login`** — You sign in with your Anthropic account; Claude Code sends your Anthropic API key as `x-api-key`.
2. **LiteLLM authentication** — You pass your LiteLLM proxy key via `ANTHROPIC_CUSTOM_HEADERS` so the proxy can authenticate and track your usage.
3. **Key forwarding** — With `forward_llm_provider_auth_headers: true`, LiteLLM forwards your `x-api-key` to Anthropic, giving it precedence over any proxy-configured keys.

## Prerequisites

- [Claude Code](https://docs.anthropic.com/en/docs/claude-code/overview) installed
- Anthropic API key (from [console.anthropic.com](https://console.anthropic.com))
- LiteLLM proxy with a virtual key for authentication

## Step 1: Configure LiteLLM Proxy

Enable forwarding of LLM provider auth headers so your Anthropic key takes precedence:

```yaml title="config.yaml"
model_list:
  - model_name: claude-sonnet-4-5
    litellm_params:
      model: anthropic/claude-sonnet-4-5
      # No api_key needed — client's key will be used

litellm_settings:
  forward_llm_provider_auth_headers: true  # Required for BYOK
```

:::info Why `forward_llm_provider_auth_headers`?

By default, LiteLLM strips `x-api-key` from client requests for security. Setting this to `true` allows client-provided provider keys (like your Anthropic key from `/login`) to be forwarded to Anthropic, overriding any proxy-configured keys.

:::

:::tip Configure via UI instead of config.yaml

You can also complete this setup from the LiteLLM admin UI:

- Add the model via **Models → Add Model**, leaving the **API Key** field blank.
- Enable the toggle at **Settings → UI Settings → "Forward LLM provider auth headers"**.

Both UI actions write to the database and override `config.yaml` at runtime.

:::

## Step 2: Create a LiteLLM Virtual Key

Create a virtual key in the LiteLLM UI or via API. 
```bash
# Example: Create key via API
curl -X POST "http://localhost:4000/key/generate" \
  -H "Authorization: Bearer sk-your-master-key" \
  -H "Content-Type: application/json" \
  -d '{"key_alias": "claude-code-byok", "models": ["claude-sonnet-4-5"]}'
```

## Step 3: Configure Claude Code

Set environment variables so Claude Code uses LiteLLM and sends your LiteLLM key for proxy auth:

```bash
# Point Claude Code to your LiteLLM proxy
export ANTHROPIC_BASE_URL="http://localhost:4000"

# Model name from your config
export ANTHROPIC_MODEL="claude-sonnet-4-5"

# LiteLLM proxy auth — this is added to every request
# Use x-litellm-api-key so the proxy authenticates you; your Anthropic key goes via x-api-key from /login
export ANTHROPIC_CUSTOM_HEADERS="x-litellm-api-key: sk-12345"
```

Replace `sk-12345` with your actual LiteLLM virtual key.

:::tip Multiple headers

For multiple headers, use newline-separated values:

```bash
export ANTHROPIC_CUSTOM_HEADERS="x-litellm-api-key: sk-12345
x-litellm-user-id: my-user-id"
```

:::

## Step 4: Sign In with Claude Code

1. Launch Claude Code:

   ```bash
   claude
   ```

2. Use **`/login`** and sign in with your Anthropic account (or use your API key directly).

3. Claude Code will send:
   - `x-api-key`: Your Anthropic API key (from `/login`)
   - `x-litellm-api-key`: Your LiteLLM key (from `ANTHROPIC_CUSTOM_HEADERS`)

4. LiteLLM authenticates you via `x-litellm-api-key`, then forwards `x-api-key` to Anthropic. Your Anthropic key takes precedence over any proxy-configured key.

## Summary

| Header | Source | Purpose |
|--------|--------|---------|
| `x-api-key` | Claude Code `/login` (Anthropic key) | Sent to Anthropic for API calls |
| `x-litellm-api-key` | `ANTHROPIC_CUSTOM_HEADERS` | Proxy authentication, tracking, rate limits |

## Troubleshooting

### Requests fail with "invalid x-api-key"

- Ensure `forward_llm_provider_auth_headers: true` is set in `litellm_settings` (or `general_settings`).
- Restart the LiteLLM proxy after changing the config.
- Verify you completed `/login` in Claude Code so your Anthropic key is being sent.

### Proxy returns 401

- Check that `ANTHROPIC_CUSTOM_HEADERS` includes `x-litellm-api-key: <your-key>`.
- Ensure the LiteLLM key is valid and has access to the model.

### Proxy key is used instead of my Anthropic key

- Confirm `forward_llm_provider_auth_headers: true` is in your config.
- The setting can be in `litellm_settings` or `general_settings` depending on your config structure.
- Enable debug logging: `LITELLM_LOG=DEBUG` to see which key is being forwarded.

## Related

- [Forward Client Headers](./../proxy/forward_client_headers.md) — Full BYOK and header forwarding docs
- [Claude Code Max Subscription](./claude_code_max_subscription.md) — Using Claude Code with OAuth/Max subscription through LiteLLM
