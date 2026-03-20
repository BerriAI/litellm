import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Vantage Integration

LiteLLM can export proxy spend data to [Vantage](https://vantage.sh) as [FOCUS 1.2](https://focus.finops.org/) formatted cost reports. This lets you visualize LLM spend alongside your cloud infrastructure costs in the Vantage dashboard.

## Overview

| Property | Details |
|----------|---------|
| Destination | Export LiteLLM usage data to Vantage Custom Provider |
| Data format | FOCUS CSV (automatically transformed from LiteLLM spend data) |
| Supported operations | Manual export, automatic scheduled export (hourly/daily/interval) |
| Authentication | Vantage API key + Custom Provider token |

## Prerequisites

You need two credentials from the [Vantage console](https://console.vantage.sh):

1. **API Key** — Go to **Settings → API Access Tokens** → Create a token with **Write** scope. The token looks like `vntg_tkn_...`.
2. **Custom Provider Token** — Go to **Settings → Integrations** → Create a **Custom Provider** integration → Copy the Provider ID (looks like `accss_crdntl_...`).

## Setup via API

The recommended setup uses the proxy admin endpoints. No config file changes needed.

### 1. Initialize credentials

```bash
curl -X POST http://localhost:4000/vantage/init \
  -H "Authorization: Bearer $LITELLM_ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "api_key": "vntg_tkn_YOUR_VANTAGE_API_KEY",
    "integration_token": "accss_crdntl_YOUR_PROVIDER_TOKEN"
  }'
```

Credentials are encrypted and stored in the proxy database.

### 2. Preview data (dry run)

```bash
curl -X POST http://localhost:4000/vantage/dry-run \
  -H "Authorization: Bearer $LITELLM_ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{"limit": 10}'
```

This returns FOCUS-transformed data without sending anything to Vantage. Use it to verify the pipeline works and inspect the data mapping.

### 3. Export to Vantage

```bash
curl -X POST http://localhost:4000/vantage/export \
  -H "Authorization: Bearer $LITELLM_ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{}'
```

Optional parameters:
- `limit` — Max number of records to export
- `start_time_utc` / `end_time_utc` — Filter by time range (must be provided together)

### 4. Verify in Vantage

Go to **Settings → Integrations → your Custom Provider → Import Costs** tab to see uploaded CSVs. Once the status changes from "Importing and Processing" to "Stable", costs appear in **Cost Reporting → All Resources**.

## Setup via Environment Variables

For automatic scheduled exports, configure via environment variables and proxy config:

### Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `VANTAGE_API_KEY` | Yes | Vantage API access token |
| `VANTAGE_INTEGRATION_TOKEN` | Yes | Custom Provider token from Vantage dashboard |
| `VANTAGE_BASE_URL` | No | API URL override (default: `https://api.vantage.sh`) |
| `VANTAGE_EXPORT_FREQUENCY` | No | `hourly` (default), `daily`, or `interval` |
| `VANTAGE_EXPORT_INTERVAL_SECONDS` | No | Seconds between exports when frequency is `interval` |

### Proxy config

```yaml
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
      api_key: sk-your-key

litellm_settings:
  callbacks: ["vantage"]
```

```bash
export VANTAGE_API_KEY="vntg_tkn_..."
export VANTAGE_INTEGRATION_TOKEN="accss_crdntl_..."
litellm --config /path/to/config.yaml
```

The proxy registers a background job that exports data on the configured schedule.

## API Endpoints

All endpoints require admin authentication.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/vantage/init` | Store Vantage credentials (encrypted) |
| `GET` | `/vantage/settings` | View current config (credentials masked) |
| `PUT` | `/vantage/settings` | Update credentials or base URL |
| `POST` | `/vantage/dry-run` | Preview FOCUS data without uploading |
| `POST` | `/vantage/export` | Upload cost data to Vantage |
| `DELETE` | `/vantage/delete` | Remove credentials and stop scheduled exports |

## FOCUS Field Mapping

LiteLLM spend data is transformed into the FOCUS 1.2 schema:

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

## Upload Limits

Vantage enforces per-upload limits. LiteLLM handles these automatically:

- **10,000 rows** per upload — large exports are split into batches
- **2 MB** per upload — oversized batches are further split by size
- **Unsupported columns** are stripped before upload

## Related Links

- [Vantage](https://vantage.sh)
- [Vantage Custom Providers](https://docs.vantage.sh/connecting_custom_providers)
- [FOCUS Specification](https://focus.finops.org/)
- [Focus Export (S3/Parquet)](./focus.md)
