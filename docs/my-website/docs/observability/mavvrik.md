import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Mavvrik Integration

LiteLLM provides an integration with Mavvrik, allowing you to export your LLM usage data to Mavvrik for AI cost tracking and analysis.

## Overview

| Property | Details |
|----------|---------|
| Description | Export LiteLLM daily usage data to Mavvrik via GCS signed URL uploads |
| callback name | `mavvrik` |
| Supported Operations | Automatic daily data export, manual data export, dry run testing, cost and token usage tracking |
| Data Format | CSV (gzip-compressed) uploaded to GCS via Mavvrik-issued signed URLs |
| Export Frequency | Hourly scheduler check — exports complete calendar days (never today's partial data) |

## Environment Variables

| Variable | Required | Description | Example |
|----------|----------|-------------|---------|
| `MAVVRIK_API_KEY` | Yes | Your Mavvrik API key (`x-api-key` header) | `mav_xxxxxxxxxx` |
| `MAVVRIK_API_ENDPOINT` | Yes | Mavvrik API base URL including your tenant path | `https://api.mavvrik.dev/my-tenant` |
| `MAVVRIK_CONNECTION_ID` | Yes | Connection/instance ID assigned by Mavvrik | `litellm-prod` |
| `MAVVRIK_TIMEZONE` | No | Timezone for date handling (default: `UTC`) | `America/New_York` |
| `MAVVRIK_EXPORT_INTERVAL_MINUTES` | No | Scheduler check frequency in minutes (default: `60`) | `60` |

## Setup

### Step 1: Configure Environment Variables

Set your Mavvrik credentials in your environment:

```bash
export MAVVRIK_API_KEY="mav_xxxxxxxxxx"
export MAVVRIK_API_ENDPOINT="https://api.mavvrik.dev/my-tenant"
export MAVVRIK_CONNECTION_ID="litellm-prod"
export MAVVRIK_TIMEZONE="UTC"  # Optional, defaults to UTC
```

### Step 2: Enable Mavvrik Integration

Add the Mavvrik callback to your LiteLLM configuration YAML file:

```yaml
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
      api_key: sk-xxxxxxx

litellm_settings:
  callbacks: ["mavvrik"]  # Enable Mavvrik integration
```

### Step 3: Start LiteLLM Proxy

Start your LiteLLM proxy with the configuration:

```bash
litellm --config /path/to/config.yaml
```

LiteLLM will automatically register with the Mavvrik API on startup and begin scheduling hourly exports. The initial export window is determined by the `metricsMarker` returned by Mavvrik during registration.

## Alternative Setup: API-Based Initialization

If you prefer to configure Mavvrik without restarting the proxy, use the `/mavvrik/init` endpoint:

```bash
curl -X POST "http://localhost:4000/mavvrik/init" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-admin-key" \
  -d '{
    "api_key": "mav_xxxxxxxxxx",
    "api_endpoint": "https://api.mavvrik.dev/my-tenant",
    "connection_id": "litellm-prod",
    "timezone": "UTC"
  }' | jq
```

**Expected Response:**
```json
{
  "message": "Mavvrik settings initialized successfully",
  "status": "success"
}
```

This stores encrypted credentials in the database and registers the background export job immediately — no proxy restart required.

## Testing Your Setup

### Dry Run Export

Preview the CSV records that would be uploaded for a given date without sending any data to GCS:

```bash
curl -X POST "http://localhost:4000/mavvrik/dry-run" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-admin-key" \
  -d '{
    "limit": 10
  }' | jq
```

**Expected Response:**
```json
{
  "message": "Mavvrik dry run completed",
  "status": "success",
  "dry_run_data": {
    "usage_data": [...],
    "csv_preview": "date,model,team_id,user,spend,prompt_tokens,completion_tokens,..."
  },
  "summary": {
    "total_records": 10,
    "total_cost": 0.05,
    "total_tokens": 1250,
    "unique_models": 3,
    "unique_teams": 2
  }
}
```

### Manual Export

Trigger an immediate upload to GCS for a specific date:

```bash
curl -X POST "http://localhost:4000/mavvrik/export" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-admin-key" \
  -d '{
    "date_str": "2024-01-15",
    "limit": 100
  }' | jq
```

**Expected Response:**
```json
{
  "message": "Mavvrik export completed successfully for 2024-01-15",
  "status": "success",
  "records_exported": 87
}
```

Omitting `date_str` defaults to yesterday. Re-exporting the same date overwrites the existing GCS file — exports are idempotent.

### View Current Settings

Check the current Mavvrik configuration (API key is masked):

```bash
curl -X GET "http://localhost:4000/mavvrik/settings" \
  -H "Authorization: Bearer sk-admin-key" | jq
```

**Expected Response:**
```json
{
  "api_key_masked": "mav_****xxxx",
  "api_endpoint": "https://api.mavvrik.dev/my-tenant",
  "connection_id": "litellm-prod",
  "timezone": "UTC",
  "marker": "2024-01-14",
  "status": "configured"
}
```

The `marker` field shows the last successfully exported date. The next scheduled run will export from `(marker + 1 day)` onwards.

## Data Export Details

### Export Schedule

- **Frequency**: Scheduler runs every 60 minutes (configurable via `MAVVRIK_EXPORT_INTERVAL_MINUTES`)
- **Scope**: Each run exports all complete calendar days since the last marker — never today's partial data
- **Catch-up**: If the proxy was offline for multiple days, the scheduler automatically back-fills all missed days on the next run
- **Idempotency**: Each day's data is uploaded to a GCS object named by date (e.g. `2024-01-15`). Re-exporting the same date safely overwrites the previous file

### Upload Flow

Each export follows a 3-step GCS resumable upload pattern:

1. **GET signed URL** — LiteLLM requests a time-limited GCS signed URL from Mavvrik
2. **POST initiate** — LiteLLM initiates a resumable upload session with GCS
3. **PUT payload** — LiteLLM uploads the gzip-compressed CSV to GCS

### Data Format

LiteLLM exports daily spend aggregates from `LiteLLM_DailyUserSpend` as a CSV file. Each row represents one model/team/user combination for a given day and includes:

| Column | Description |
|--------|-------------|
| `date` | Calendar date (YYYY-MM-DD) |
| `model` | LLM model name |
| `team_id` | LiteLLM team identifier |
| `user` | LiteLLM user identifier |
| `spend` | Total cost in USD |
| `prompt_tokens` | Input tokens consumed |
| `completion_tokens` | Output tokens generated |
| `successful_requests` | Count of successful API calls |
| `connection_id` | Your Mavvrik connection ID (added by LiteLLM) |

Only rows with `successful_requests > 0` are exported — zero-request rows are skipped.

## Advanced Configuration

### Reset Export Cursor

If Mavvrik asks you to re-export from an earlier date (e.g. after a data reset), update the marker via the settings endpoint:

```bash
curl -X PUT "http://localhost:4000/mavvrik/settings" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-admin-key" \
  -d '{
    "marker": "2024-01-01"
  }' | jq
```

The next scheduled run will export from `2024-01-02` onwards, re-uploading all days since that date.

### Custom Export Frequency

Change how often the scheduler checks for new days to export:

```bash
export MAVVRIK_EXPORT_INTERVAL_MINUTES=120  # Check every 2 hours
```

### Remove Mavvrik Integration

Delete all Mavvrik settings and deregister the background job:

```bash
curl -X DELETE "http://localhost:4000/mavvrik/delete" \
  -H "Authorization: Bearer sk-admin-key" | jq
```

## Troubleshooting

### Common Issues

1. **`status: not_configured` in logs**
   ```
   Initialized Success Callbacks - ['mavvrik']
   ```
   Ensure `MAVVRIK_API_KEY`, `MAVVRIK_API_ENDPOINT`, and `MAVVRIK_CONNECTION_ID` are all set in the environment. The integration auto-initializes from these env vars when the proxy starts.

2. **401 on registration**
   ```
   Mavvrik registration failed: 401 Unauthorized
   ```
   Verify your `MAVVRIK_API_KEY` is valid and that `MAVVRIK_API_ENDPOINT` includes your tenant path (e.g. `https://api.mavvrik.dev/my-tenant`, not just `https://api.mavvrik.dev`).

3. **No data appearing in Mavvrik**
   - Use the dry-run endpoint to verify data exists for the target date
   - Check the `marker` field in `GET /mavvrik/settings` — if it is already at yesterday's date, there is nothing new to export
   - Ensure the proxy has been generating traffic (check `LiteLLM_DailyUserSpend` table)
   - Only complete calendar days are exported — today's data will appear after midnight UTC

4. **Missing credentials error on export**
   ```
   MavvrikLogger: missing required config fields: ['api_key']
   ```
   Either set the `MAVVRIK_*` environment variables or call `POST /mavvrik/init` to store credentials in the database.

5. **Export succeeds but `records_exported: 0`**
   All rows for that date had `successful_requests == 0`. This is normal if the proxy received requests that day but all failed.

## Related Links

- [Mavvrik Documentation](https://help.mavvrik.ai/)
