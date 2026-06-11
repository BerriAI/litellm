# Mavvrik Integration

LiteLLM can export proxy spend data to [Mavvrik](https://mavvrik.ai) as [FOCUS 1.2](https://focus.finops.org/) formatted cost reports. This lets you track and analyse LLM spend within the Mavvrik cost management platform.

## Overview

| Property | Details |
|----------|---------|
| Destination | Export LiteLLM usage data to Mavvrik via signed URL upload |
| Data format | FOCUS CSV (gzip-compressed, automatically transformed from LiteLLM spend data) |
| Supported operations | Automatic daily export |
| Authentication | Mavvrik API key + connection ID |

## Prerequisites

You need the following from your Mavvrik account:

1. **API Key** — available in the Mavvrik dashboard under Settings.
2. **API Endpoint** — your tenant-specific API URL, e.g. `https://api.mavvrik.ai/<tenant_id>`.
3. **Connection ID** — the AI cost connection ID configured in your Mavvrik account.

## Setup

### Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `MAVVRIK_API_KEY` | Yes | Mavvrik API key |
| `MAVVRIK_API_ENDPOINT` | Yes | Tenant API endpoint, e.g. `https://api.mavvrik.ai/<tenant_id>` |
| `MAVVRIK_CONNECTION_ID` | Yes | AI cost connection identifier |
| `MAVVRIK_FOCUS_MAX_ROWS` | No | Maximum rows per daily export (default: 500000). Increase for very high-traffic deployments. |

:::info Daily export only
Only `daily` frequency is supported. The Mavvrik ingestion protocol stores one file per calendar date (`metrics/YYYY-MM-DD`). Hourly or interval exports would overwrite each other within the same day, producing incomplete data.
:::

### Proxy config

```yaml
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
      api_key: sk-your-key

litellm_settings:
  callbacks: ["mavvrik"]
```

```bash
export MAVVRIK_API_KEY="<your-api-key>"
export MAVVRIK_API_ENDPOINT="https://api.mavvrik.ai/<tenant_id>"
export MAVVRIK_CONNECTION_ID="<connection-id>"
litellm --config /path/to/config.yaml
```

The proxy registers a background job that exports FOCUS-formatted spend data once daily.

## How it works

Each daily export cycle:

1. Registers the connector with the Mavvrik backend (`POST /metrics/agent/ai/{connection_id}`)
2. Requests a GCS signed upload URL for the export date (`GET /metrics/agent/ai/{connection_id}/upload-url`)
3. Uploads gzip-compressed FOCUS CSV to GCS via the signed URL

Re-running an export for the same date overwrites the previous file — exports are idempotent. Exports are capped at `MAVVRIK_FOCUS_MAX_ROWS` rows per day (default 500k) to bound memory usage.

## FOCUS Field Mapping

LiteLLM spend data is transformed into the FOCUS 1.2 schema before upload:

| LiteLLM Field | FOCUS Column | Description |
|---------------|-------------|-------------|
| `spend` | BilledCost, EffectiveCost | Cost of the usage |
| `model` | ChargeDescription, ResourceId | Model identifier |
| `model_group` | ServiceName | Model group / deployment |
| `custom_llm_provider` | ProviderName, PublisherName | Provider (openai, anthropic, etc.) |
| `api_key` | BillingAccountId | Hashed API key |
| `api_key_alias` | BillingAccountName | Human-readable key alias |
| `team_id` | SubAccountId | Team identifier |
| `team_alias` | SubAccountName | Team name |

Additional metadata (user_id, model_group, etc.) is included in the `Tags` column as JSON.

## Related Links

- [Mavvrik](https://mavvrik.ai)
- [FOCUS Specification](https://focus.finops.org/)
- [Focus Export (S3/GCS)](./focus.md)
