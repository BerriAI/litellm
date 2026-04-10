# MCP Token Validation

Enforce that per-user OAuth tokens belong to your corporate workspace before LiteLLM stores them.

**Use case:** Your company uses Slack. You want to ensure employees can only connect their `@yourco.com` Slack account — not a personal one. Configure `token_validation` rules on the MCP server, and LiteLLM rejects any token whose claims don't match.

## How it works

When a user completes OAuth and submits their token via `POST /server/{id}/oauth-user-credential`, LiteLLM checks the `token_response_metadata` fields against the `token_validation` rules you've configured. If any field is missing or doesn't match, the request is rejected with a `403` before the credential is stored.

## Configuration

### Via the UI

Open **MCP Servers → Add New MCP Server**, set auth type to **OAuth**, and scroll to **Required Token Claims**:

![Token claims builder](/img/mcp_token_validation.png)

Click a quick-add chip (Slack enterprise, Jira cloud, GitHub Enterprise) or type your own key. Fill in the required value.

### Via API

```bash
curl -X POST https://your-proxy/v1/mcp/server \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "slack-corp",
    "url": "https://slack.com/api/mcp",
    "transport": "http",
    "auth_type": "oauth2",
    "token_validation": {
      "enterprise_id": "E04COREWEAVE"
    }
  }'
```

### Via config.yaml

```yaml title="config.yaml"
mcp_servers:
  slack_corp:
    url: "https://slack.com/api/mcp"
    auth_type: oauth2
    token_validation:
      enterprise_id: "E04COREWEAVE"
```

## Expected behavior

When a user submits their OAuth credential:

| Scenario | Result |
|----------|--------|
| `token_response_metadata: {"enterprise_id": "E04COREWEAVE"}` | ✅ 200 — credential stored |
| `token_response_metadata: {"enterprise_id": "E99PERSONAL"}` | ❌ 403 — `'enterprise_id' = 'E99PERSONAL', expected 'E04COREWEAVE'` |
| `token_response_metadata: {}` (field absent) | ❌ 403 — `required field 'enterprise_id' is absent` |

## Common claim keys

| Provider | Key | Example value |
|----------|-----|---------------|
| Slack | `enterprise_id` | `E04XXXXXXX` |
| Jira / Confluence | `cloud_id` | `abc-123-def` |
| GitHub Enterprise | `enterprise_id` | `your-org` |

## Token Storage TTL

Optionally set how long credentials are cached (in seconds):

```json
{
  "token_validation": {"enterprise_id": "E04COREWEAVE"},
  "token_storage_ttl_seconds": 3600
}
```
