# Auto Sync Anthropic Beta Headers

Automatically keep your Anthropic beta headers configuration up to date without restarting your service. **This allows you to support new Anthropic beta features across all providers without restarting your service.**

## Overview

When Anthropic releases new beta features (e.g., new tool capabilities, extended context windows), you typically need to restart your LiteLLM service to get the latest beta header mappings for different providers (Anthropic, Bedrock, Vertex AI, Azure AI).

With auto-sync, LiteLLM automatically pulls the latest configuration from GitHub's [`anthropic_beta_headers_config.json`](https://github.com/BerriAI/litellm/blob/main/litellm/anthropic_beta_headers_config.json) without requiring a restart. This means:

- **Zero downtime** when new beta features are released
- **Always up-to-date** provider support mappings
- **Automatic updates** - set it once and forget it

## Quick Start

**Manual sync:**
```bash
curl -X POST "https://your-proxy-url/reload/anthropic_beta_headers" \
  -H "Authorization: Bearer YOUR_ADMIN_TOKEN" \
  -H "Content-Type: application/json"
```

**Automatic sync every 24 hours:**
```bash
curl -X POST "https://your-proxy-url/schedule/anthropic_beta_headers_reload?hours=24" \
  -H "Authorization: Bearer YOUR_ADMIN_TOKEN" \
  -H "Content-Type: application/json"
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reload/anthropic_beta_headers` | POST | Manual sync |
| `/schedule/anthropic_beta_headers_reload?hours={hours}` | POST | Schedule periodic sync |
| `/schedule/anthropic_beta_headers_reload` | DELETE | Cancel scheduled sync |
| `/schedule/anthropic_beta_headers_reload/status` | GET | Check sync status |

**Authentication:** Requires admin role or master key

## Python Example

```python
import requests

def sync_anthropic_beta_headers(proxy_url, admin_token):
    response = requests.post(
        f"{proxy_url}/reload/anthropic_beta_headers",
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    return response.json()

# Usage
result = sync_anthropic_beta_headers("https://your-proxy-url", "your-admin-token")
print(result['message'])
```

## Configuration

**Custom beta headers config URL:**
```bash
export LITELLM_ANTHROPIC_BETA_HEADERS_URL="https://raw.githubusercontent.com/BerriAI/litellm/main/litellm/anthropic_beta_headers_config.json"
```

**Use local beta headers config:**
```bash
export LITELLM_LOCAL_ANTHROPIC_BETA_HEADERS=True
```

## Scheduling Automatic Reloads

Schedule automatic reloads to ensure your proxy always has the latest beta header mappings:

```bash
# Reload every 24 hours
curl -X POST "https://your-proxy-url/schedule/anthropic_beta_headers_reload?hours=24" \
  -H "Authorization: Bearer YOUR_ADMIN_TOKEN"
```

**Check reload status:**
```bash
curl -X GET "https://your-proxy-url/schedule/anthropic_beta_headers_reload/status" \
  -H "Authorization: Bearer YOUR_ADMIN_TOKEN"
```

**Response:**
```json
{
  "scheduled": true,
  "interval_hours": 24,
  "last_run": "2026-02-13T10:00:00",
  "next_run": "2026-02-14T10:00:00"
}
```

**Cancel scheduled reload:**
```bash
curl -X DELETE "https://your-proxy-url/schedule/anthropic_beta_headers_reload" \
  -H "Authorization: Bearer YOUR_ADMIN_TOKEN"
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LITELLM_ANTHROPIC_BETA_HEADERS_URL` | URL to fetch beta headers config from | GitHub main branch |
| `LITELLM_LOCAL_ANTHROPIC_BETA_HEADERS` | Set to `True` to use local config only | `False` |

## How It Works

1. **Initial Load:** On startup, LiteLLM loads the beta headers configuration from the remote URL (or local file if configured)
2. **Caching:** The configuration is cached in memory to avoid repeated fetches on every request
3. **Scheduled Reload:** If configured, the proxy checks every 10 seconds whether it's time to reload based on your schedule
4. **Manual Reload:** You can trigger an immediate reload via the API endpoint
5. **Multi-Pod Support:** In multi-pod deployments, the reload configuration is stored in the database so all pods stay in sync

## Benefits

- **No Restarts Required:** Add support for new Anthropic beta features without downtime
- **Provider Compatibility:** Automatically get updated mappings for Bedrock, Vertex AI, Azure AI, etc.
- **Performance:** Configuration is cached and only reloaded when needed
- **Reliability:** Falls back to local configuration if remote fetch fails

## Related

- [Model Cost Map Sync](./sync_models_github.md) - Auto-sync model pricing data
- [Anthropic Beta Headers](../completion/anthropic.md#beta-features) - Using Anthropic beta features
