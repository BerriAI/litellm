# Auto Sync New Models (Day-0 Launches)

Automatically keep your model pricing and context window data up to date without restarting your service. **This allows you to add day-0 support for new models without restarting your service.**

## Overview

When providers like OpenAI or Anthropic release new models (e.g., GPT-5, Claude 4), you typically need to restart your LiteLLM service to get the latest pricing and context window data. 

With auto-sync, LiteLLM automatically pulls the latest model data from GitHub's [`model_prices_and_context_window.json`](https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json) without requiring a restart. This means:

- **Zero downtime** when new models are released
- **Always accurate pricing** for cost tracking and budgets
- **Automatic updates** - set it once and forget it

<iframe width="840" height="500" src="https://www.loom.com/embed/ba41acc1882d41b284bbddbb0e9c27ce?sid=bdae351e-2026-4e39-932b-fcb185ff612c" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen></iframe>

<br/>
<br/>

## Quick Start

**Manual sync:**
```bash
curl -X POST "https://your-proxy-url/reload/model_cost_map" \
  -H "Authorization: Bearer YOUR_ADMIN_TOKEN" \
  -H "Content-Type: application/json"
```

**Automatic sync every 6 hours:**
```bash
curl -X POST "https://your-proxy-url/schedule/model_cost_map_reload?hours=6" \
  -H "Authorization: Bearer YOUR_ADMIN_TOKEN" \
  -H "Content-Type: application/json"
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reload/model_cost_map` | POST | Manual sync |
| `/schedule/model_cost_map_reload?hours={hours}` | POST | Schedule periodic sync |
| `/schedule/model_cost_map_reload` | DELETE | Cancel scheduled sync |
| `/schedule/model_cost_map_reload/status` | GET | Check sync status |

**Authentication:** Requires admin role or master key

## Python Example

```python
import requests

def sync_models(proxy_url, admin_token):
    response = requests.post(
        f"{proxy_url}/reload/model_cost_map",
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    return response.json()

# Usage
result = sync_models("https://your-proxy-url", "your-admin-token")
print(result['message'])
```

## Configuration

**Custom model cost map URL:**
```bash
export LITELLM_MODEL_COST_MAP_URL="https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
```

**Use local model cost map:**
```bash
export LITELLM_LOCAL_MODEL_COST_MAP=True
```