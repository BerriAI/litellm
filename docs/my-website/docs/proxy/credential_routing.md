import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Per-Team/Project Credential Routing

Route the same model to different LLM provider endpoints (e.g. different Azure instances) based on which team or project makes the request.

## Overview

In multi-tenant deployments, different teams often need the same model name (e.g., `gpt-4`) to hit different provider endpoints — for example, separate Azure OpenAI instances per business unit for cost isolation, data residency, or rate limit separation.

**Credential routing** lets you configure this in team/project metadata using the existing [credentials table](./ui_credentials.md), without duplicating model definitions or creating separate model groups per team.

```
Hotel Team → gpt-4 → https://hotel-eastus.openai.azure.com/
Flight Team → gpt-4 → https://flight-centralus.openai.azure.com/
```

### Precedence Chain

When a request comes in, the system walks this precedence chain (first match wins):

1. **Clientside credentials** — `api_base`/`api_key` passed in the request body ([docs](./clientside_auth.md))
2. **Project model-specific** — override for this exact model in the project's `model_config`
3. **Project default** — `defaultconfig` in the project's `model_config`
4. **Team model-specific** — override for this exact model in the team's `model_config`
5. **Team default** — `defaultconfig` in the team's `model_config`
6. **Deployment default** — the model's `litellm_params` as configured in `config.yaml`

## Quick Start

### Step 1: Create Credentials

Store your Azure endpoint credentials in the credentials table. You can do this via the [UI](./ui_credentials.md) or API:

```bash showLineNumbers
# Create credential for Hotel team's Azure endpoint
curl -X POST 'http://0.0.0.0:4000/credentials' \
-H 'Authorization: Bearer sk-1234' \
-H 'Content-Type: application/json' \
-d '{
    "credential_name": "hotel-azure-eastus",
    "credential_values": {
        "api_base": "https://hotel-eastus.openai.azure.com/",
        "api_key": "sk-azure-hotel-key-xxx"
    }
}'
```

```bash showLineNumbers
# Create credential for Flight team's Azure endpoint
curl -X POST 'http://0.0.0.0:4000/credentials' \
-H 'Authorization: Bearer sk-1234' \
-H 'Content-Type: application/json' \
-d '{
    "credential_name": "flight-azure-centralus",
    "credential_values": {
        "api_base": "https://flight-centralus.openai.azure.com/",
        "api_key": "sk-azure-flight-key-xxx"
    }
}'
```

### Step 2: Set `model_config` on Teams

Add a `model_config` key to the team's metadata referencing the credential by name:

```bash showLineNumbers
# Hotel team — default Azure endpoint for all models
curl -X PATCH 'http://0.0.0.0:4000/team/update' \
-H 'Authorization: Bearer sk-1234' \
-H 'Content-Type: application/json' \
-d '{
    "team_id": "hotel-team-id",
    "metadata": {
        "model_config": {
            "defaultconfig": {
                "azure": {
                    "litellm_credentials": "hotel-azure-eastus"
                }
            }
        }
    }
}'
```

```bash showLineNumbers
# Flight team — default Azure endpoint for all models
curl -X PATCH 'http://0.0.0.0:4000/team/update' \
-H 'Authorization: Bearer sk-1234' \
-H 'Content-Type: application/json' \
-d '{
    "team_id": "flight-team-id",
    "metadata": {
        "model_config": {
            "defaultconfig": {
                "azure": {
                    "litellm_credentials": "flight-azure-centralus"
                }
            }
        }
    }
}'
```

### Step 3: Make Requests

Requests are automatically routed to the correct Azure endpoint based on the API key's team:

```bash showLineNumbers
# Request using Hotel team's API key → routes to hotel-eastus.openai.azure.com
curl http://localhost:4000/v1/chat/completions \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer sk-hotel-team-key' \
-d '{"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]}'

# Request using Flight team's API key → routes to flight-centralus.openai.azure.com
curl http://localhost:4000/v1/chat/completions \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer sk-flight-team-key' \
-d '{"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]}'
```

## Per-Model Overrides

You can set different credentials for specific models while keeping a default for everything else:

```bash showLineNumbers
curl -X PATCH 'http://0.0.0.0:4000/team/update' \
-H 'Authorization: Bearer sk-1234' \
-H 'Content-Type: application/json' \
-d '{
    "team_id": "hotel-team-id",
    "metadata": {
        "model_config": {
            "defaultconfig": {
                "azure": {
                    "litellm_credentials": "hotel-azure-eastus"
                }
            },
            "gpt-4": {
                "azure": {
                    "litellm_credentials": "hotel-azure-westus"
                }
            }
        }
    }
}'
```

With this config:
- `gpt-4` requests → `hotel-azure-westus` credential (model-specific)
- All other models → `hotel-azure-eastus` credential (default)

## Project-Level Overrides

Projects inherit their team's `model_config` but can override at the project level. Project overrides take precedence over team overrides.

```bash showLineNumbers
# Project overrides the team default for all models
curl -X PATCH 'http://0.0.0.0:4000/project/update' \
-H 'Authorization: Bearer sk-1234' \
-H 'Content-Type: application/json' \
-d '{
    "project_id": "hotel-rec-app-id",
    "metadata": {
        "model_config": {
            "defaultconfig": {
                "azure": {
                    "litellm_credentials": "hotel-rec-azure"
                }
            },
            "gpt-4-vision": {
                "azure": {
                    "litellm_credentials": "hotel-rec-vision"
                }
            }
        }
    }
}'
```

### Full Example: Hotel Team with Two Projects

**Setup:**
- **Hotel Team**: default `hotel-azure-eastus`, GPT-4 override to `hotel-azure-westus`
- **Hotel Rec App** (project): default `hotel-rec-azure`, GPT-4-Vision override to `hotel-rec-vision`
- **Hotel Review App** (project): no overrides — inherits team config

**Resolution:**

| Request | Resolved Credential | Why |
|---|---|---|
| Hotel Rec App → `gpt-4` | `hotel-rec-azure` | Project default (no project model-specific match for gpt-4) |
| Hotel Rec App → `gpt-4-vision` | `hotel-rec-vision` | Project model-specific |
| Hotel Review App → `gpt-3.5` | `hotel-azure-eastus` | Team default (no project config) |
| Hotel Review App → `gpt-4` | `hotel-azure-westus` | Team model-specific |

## `model_config` Schema

The `model_config` key is a JSON object in team/project `metadata`:

```json
{
    "model_config": {
        "defaultconfig": {
            "<provider>": {
                "litellm_credentials": "<credential-name>"
            }
        },
        "<model-name>": {
            "<provider>": {
                "litellm_credentials": "<credential-name>"
            }
        }
    }
}
```

| Field | Description |
|---|---|
| `defaultconfig` | Fallback credential for any model not explicitly listed |
| `<model-name>` | Model-specific override — must match the LiteLLM model group name |
| `<provider>` | Provider key (e.g. `azure`, `openai`, `bedrock`). When the model name includes a provider prefix (e.g. `azure/gpt-4`), the system prefers the matching provider key |
| `litellm_credentials` | Name of a credential in the [credentials table](./ui_credentials.md) |

### Credential Values

The referenced credential can contain any combination of:

| Key | Description |
|---|---|
| `api_base` | Provider endpoint URL |
| `api_key` | API key for the provider |
| `api_version` | API version (e.g. for Azure) |

Only keys present in the credential are applied. Keys already in the request (e.g. clientside `api_version`) are never overwritten.

## Enabling the Feature

This feature is **disabled by default** and must be explicitly enabled. To enable it:

<Tabs>

<TabItem value="config" label="config.yaml">

```yaml
litellm_settings:
    enable_model_config_credential_overrides: true
```

</TabItem>

<TabItem value="env" label="Environment Variable">

```bash
export LITELLM_ENABLE_MODEL_CONFIG_CREDENTIAL_OVERRIDES=true
```

</TabItem>

</Tabs>

:::info
The feature flag must be enabled before `model_config` entries in team/project metadata take effect. Without it, credential routing is completely inert — no metadata is read, no credentials are resolved.
:::

## Related Documentation

- [Adding LLM Credentials](./ui_credentials.md) — Create and manage reusable credentials
- [Project Management](./project_management.md) — Project hierarchy and API
- [Team Budgets](./team_budgets.md) — Team-level budget management
- [Clientside LLM Credentials](./clientside_auth.md) — Passing credentials in the request body
- [Credential Usage Tracking](./credential_usage_tracking.md) — Track spend by credential
