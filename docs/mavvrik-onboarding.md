# Mavvrik Integration — User Onboarding Guide

This guide walks through setting up the Mavvrik cost-data integration with
a self-hosted LiteLLM proxy. After completing these steps, LiteLLM will
automatically upload one CSV file per day to Mavvrik's GCS bucket.

---

## Prerequisites

- LiteLLM proxy running with a PostgreSQL database
- LiteLLM proxy master key (used for admin API calls)
- Mavvrik credentials:
  - **API key** (`x-api-key` value)
  - **API endpoint** (e.g. `https://api.mavvrik.dev`)
  - **Tenant slug** (your Mavvrik tenant identifier)
  - **Instance ID** (a unique identifier for this LiteLLM deployment)

---

## Step 1 — Initialize Mavvrik Settings

Call `/mavvrik/init` once to store your credentials and register this
LiteLLM instance with Mavvrik. This call also fetches the initial
`metricsMarker` from Mavvrik (the date from which it wants data).

```bash
curl -X POST https://your-litellm-proxy/mavvrik/init \
  -H "Authorization: Bearer <your-litellm-master-key>" \
  -H "Content-Type: application/json" \
  -d '{
    "api_key":      "<mavvrik-api-key>",
    "api_endpoint": "https://api.mavvrik.dev",
    "tenant":       "<your-tenant-slug>",
    "instance_id":  "<your-instance-id>",
    "timezone":     "UTC"
  }'
```

**Expected response:**
```json
{ "message": "Mavvrik settings initialized successfully", "status": "success" }
```

What happens behind the scenes:
1. LiteLLM calls the Mavvrik register endpoint to get the initial `metricsMarker`
2. Credentials are encrypted and stored in the `LiteLLM_Config` database table
3. The `metricsMarker` is converted to a date string and stored as the initial marker

> **Note:** The register call is best-effort. If Mavvrik is temporarily
> unavailable, settings are still saved and the marker defaults to the
> first day of the current month.

---

## Step 2 — Verify Settings

Confirm the settings were saved and the initial marker was received:

```bash
curl -H "Authorization: Bearer <your-litellm-master-key>" \
  https://your-litellm-proxy/mavvrik/settings
```

**Expected response:**
```json
{
  "api_key_masked": "czqA...kum",
  "api_endpoint":   "https://api.mavvrik.dev",
  "tenant":         "my-tenant",
  "instance_id":    "litellm-prod",
  "timezone":       "UTC",
  "marker":         "2026-02-01",
  "status":         "configured"
}
```

The `marker` field is a date string (`YYYY-MM-DD`) representing the last
successfully exported day. The next scheduled run will export from
`marker + 1 day` up to yesterday.

If `marker` is `null`, the first export will automatically start from the earliest
date that has rows in `LiteLLM_DailyUserSpend` — all historical data is picked up
without any manual backfill.

---

## Step 3 — Preview Data (Dry Run)

Before uploading anything, preview the records that would be sent.
The dry-run always shows yesterday's data — the most recent complete day.

```bash
curl -X POST https://your-litellm-proxy/mavvrik/dry-run \
  -H "Authorization: Bearer <your-litellm-master-key>" \
  -H "Content-Type: application/json" \
  -d '{"limit": 20}'
```

**Expected response:**
```json
{
  "status": "success",
  "message": "Mavvrik dry run completed",
  "dry_run_data": {
    "usage_data":  [...],       // first 20 raw spend rows from yesterday
    "csv_preview": "id,date,user_id,api_key,model,...\n..."
  },
  "summary": {
    "total_records":  63,
    "total_cost":     12.34,
    "total_tokens":   450000,
    "unique_models":  3,
    "unique_teams":   3
  }
}
```

The `csv_preview` shows the first 5000 characters of the CSV that would be
uploaded to GCS. Each row is one spend record from `LiteLLM_DailyUserSpend`
with team/key metadata joined in.

---

## Step 4 — Enable the Scheduled Export

Edit your LiteLLM proxy config YAML to enable the Mavvrik callback:

```yaml
# litellm-config.yaml
litellm_settings:
  callbacks: ["mavvrik"]
```

Then restart the proxy. On startup, LiteLLM will:
1. Detect that `mavvrik_settings` exists in the database
2. Register an hourly APScheduler job
3. On each run, call the Mavvrik register endpoint then export all complete days since the marker

> **How exports work:**
> - Each run starts by calling the Mavvrik register endpoint (POST to the agent path)
>   to verify connectivity and retrieve Mavvrik's current `metricsMarker`. If Mavvrik's
>   marker is earlier than the local one (e.g. Mavvrik reset their cursor), LiteLLM
>   honours Mavvrik's date and re-exports from that point automatically. If the call
>   fails, a warning is logged and the export continues with the local marker.
> - After the register check, each run covers all dates from `(marker + 1 day)` up to **yesterday**
> - Today is never exported — spend rows are still accumulating
> - Each day creates one GCS file named `YYYY-MM-DD`
> - Re-uploading the same date overwrites the file (idempotent)

---

## Step 5 — Trigger a Manual Export (Optional)

To export a specific day immediately without waiting for the scheduler:

```bash
# Export yesterday (default)
curl -X POST https://your-litellm-proxy/mavvrik/export \
  -H "Authorization: Bearer <your-litellm-master-key>" \
  -H "Content-Type: application/json" \
  -d '{}'
```

**Expected response:**
```json
{ "message": "Mavvrik export completed successfully for 2026-02-17", "status": "success" }
```

### Export a specific date

```bash
curl -X POST https://your-litellm-proxy/mavvrik/export \
  -H "Authorization: Bearer <your-litellm-master-key>" \
  -H "Content-Type: application/json" \
  -d '{"date_str": "2026-01-15"}'
```

Manual exports do **not** advance the scheduled marker. Re-exporting the same
date is safe — the GCS file is overwritten with the latest data.

### Backfill a date range

```bash
for date in 2026-01-01 2026-01-02 2026-01-03; do
  curl -X POST https://your-litellm-proxy/mavvrik/export \
    -H "Authorization: Bearer <your-litellm-master-key>" \
    -H "Content-Type: application/json" \
    -d "{\"date_str\": \"${date}\"}"
  sleep 2
done
```

---

## Step 6 — Confirm Data in Mavvrik

After a successful export, the GCS object appears at:
```
{bucket}/{tenant}/k8s/{instance_id}/metrics/{date}

Example:
  acme-bucket/my-tenant/k8s/litellm-prod/metrics/2026-02-17
```

One file per calendar day. Verify in the Mavvrik dashboard that cost data is
flowing under your tenant.

---

## Configuration Reference

### Required (set via `/mavvrik/init`)

| Field | Description |
|-------|-------------|
| `api_key` | Mavvrik API key (sent as `x-api-key` header) |
| `api_endpoint` | Mavvrik API base URL (e.g. `https://api.mavvrik.dev`) |
| `tenant` | Your Mavvrik tenant slug |
| `instance_id` | Unique identifier for this LiteLLM deployment |
| `timezone` | Timezone for logging (default: `UTC`) |

### Optional (environment variables)

| Variable | Default | Description |
|----------|---------|-------------|
| `MAVVRIK_EXPORT_INTERVAL_MINUTES` | `60` | How often the scheduler checks for new days to export |
| `MAVVRIK_MAX_FETCHED_DATA_RECORDS` | `50000` | Max rows fetched per day export |

---

## Admin Endpoints Summary

All endpoints require `PROXY_ADMIN` role (pass master key as Bearer token).

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `POST` | `/mavvrik/init` | Store credentials, register with Mavvrik, set initial marker |
| `GET` | `/mavvrik/settings` | View current config (API key masked) + current marker date |
| `PUT` | `/mavvrik/settings` | Update one or more settings fields (including marker reset) |
| `POST` | `/mavvrik/dry-run` | Preview yesterday's CSV records without uploading |
| `POST` | `/mavvrik/export` | Export a specific date to GCS (default: yesterday) |

---

## Troubleshooting

### `"Mavvrik not configured. Call POST /mavvrik/init first."`
Run Step 1. Settings are not in the database.

### `"DB not connected"`
The LiteLLM proxy is not connected to PostgreSQL. Check `DATABASE_URL`.

### `"Mavvrik signed URL request failed: 404"`
The upload-url endpoint returned 404. Check that `tenant` and `instance_id`
match what Mavvrik expects for your account.

### `"Mavvrik registration failed: 406"`
The register payload doesn't match what Mavvrik expects.
The integration sends the k8s-appliance format — contact Mavvrik support
if this persists.

### `"Mavvrik GCS initiate upload failed: 403 SignatureDoesNotMatch"`
The signed URL content type doesn't match. This is a server-side issue
with the Mavvrik URL generation — contact Mavvrik support.

### Export runs but no data appears in Mavvrik
- Run `/mavvrik/dry-run` to confirm there is data in the DB for yesterday
- Check that `callbacks: ["mavvrik"]` is in the proxy config
- Check proxy logs for scheduler errors: `grep -i mavvrik /path/to/proxy.log`
- Confirm the `marker` date via `GET /mavvrik/settings` — if it is already
  at yesterday, no new export will run until tomorrow

### How to re-export a specific day
```bash
curl -X POST https://your-litellm-proxy/mavvrik/export \
  -H "Authorization: Bearer <your-litellm-master-key>" \
  -H "Content-Type: application/json" \
  -d '{"date_str": "2026-01-15"}'
```
This overwrites the existing GCS file for that date with the latest data.
It does not affect the scheduled cursor.

### How to reset the marker (re-export from a specific date onwards)
```bash
curl -X PUT https://your-litellm-proxy/mavvrik/settings \
  -H "Authorization: Bearer <your-litellm-master-key>" \
  -H "Content-Type: application/json" \
  -d '{"marker": "2026-01-31"}'
```
After this, the next scheduled run will export Feb 1 onwards automatically.
