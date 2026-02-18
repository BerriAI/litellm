# Mavvrik Integration — Data Flow Reference

This document explains how cost data flows from LiteLLM into Mavvrik's GCS bucket,
how the date-based export cursor works, and how past/future data is handled.

---

## Overview

```
LiteLLM Proxy
  │  (every API call)
  ▼
LiteLLM_DailyUserSpend table  ← one row per (user, date, key, model, provider) per day
  │                               rows accumulate throughout the day, never replaced
  │  (every 60 min, scheduled)
  ▼
Mavvrik Export Job
  ├── For each missed date from (marker + 1 day) up to yesterday:
  │     ├── Query all rows where dus.date = that date
  │     ├── Write as CSV (all columns, gzip-compressed)
  │     ├── GET signed URL from Mavvrik API  (?name=YYYY-MM-DD)
  │     ├── POST to GCS (initiate resumable upload)
  │     ├── PUT to GCS (upload gzip-compressed CSV)
  │     ├── Advance marker to that date in LiteLLM_Config
  │     └── PATCH Mavvrik agent endpoint (metricsMarker epoch)
  └── Today is never exported (rows still accumulating)
```

---

## Source Table: `LiteLLM_DailyUserSpend`

LiteLLM upserts one row per `(user_id, date, api_key, model, provider)` combination
after every completion call. Multiple API calls within the same day are **aggregated**
into a single row — tokens and spend are summed, not duplicated.

The key field for export filtering is **`date`** (YYYY-MM-DD). Each export covers
exactly one complete calendar day.

### How rows are written

- Every successful API call → enqueued in memory
- Every ~10–15 seconds → batch flushed to DB via Prisma upsert
- `updated_at` is set by Prisma on each batch flush — not per API call

This means by the time an export runs (hourly), all spend for previous days is
fully settled. Today's rows are excluded because they are still accumulating.

### Columns exported (all columns from the query)

| Column | Description |
|--------|-------------|
| `id` | Row UUID |
| `date` | Spend date (YYYY-MM-DD) — used for export filtering |
| `user_id` | LiteLLM user identifier |
| `api_key` | Hashed virtual key |
| `model` | LLM model name (e.g. `gpt-4o`) |
| `model_group` | Model family/group |
| `custom_llm_provider` | Provider (e.g. `openai`, `anthropic`) |
| `prompt_tokens` | Input tokens |
| `completion_tokens` | Output tokens |
| `spend` | Cost in USD |
| `api_requests` | Total requests |
| `successful_requests` | Successful requests |
| `failed_requests` | Failed requests |
| `cache_creation_input_tokens` | Cache creation tokens |
| `cache_read_input_tokens` | Cache read tokens |
| `created_at` | Row creation timestamp |
| `updated_at` | Last batch-flush timestamp |
| `team_id` | Team owning the key (from VerificationToken JOIN) |
| `api_key_alias` | Human-readable key name (from VerificationToken JOIN) |
| `team_alias` | Human-readable team name (from TeamTable JOIN) |
| `user_email` | User email (from UserTable JOIN) |

---

## The Marker — How Date Cursor Works

The **marker** is a date string (`YYYY-MM-DD`) stored in `LiteLLM_Config`
under `param_name = "mavvrik_settings"`. It records the last calendar day
that was successfully exported.

```
─────────────────────────────────────────────────────────► calendar days

  marker          yesterday   today
    │                 │         │
    ▼                 ▼         ▼
────┬─────────────────┬─────────┬──────────────────────────
    │  export each    │  last   │  never export
    │  missed day     │  day to │  (still accumulating)
    │  individually   │  export │
    └─────────────────┘
          ↓ after each day succeeds
          marker = that day's date
```

### First run (no marker)

On the very first export, no marker exists. The scheduler starts from yesterday —
the most recent complete day.

### Steady state (marker exists)

```
marker = "2026-02-15"  (last exported date)
today  = "2026-02-18"
yesterday = "2026-02-17"

Dates to export: Feb 16, Feb 17

  Export Feb 16 → upload "2026-02-16" → advance marker to "2026-02-16"
  Export Feb 17 → upload "2026-02-17" → advance marker to "2026-02-17"
  Feb 18 = today → skip
```

### Missed days (catch-up)

If the proxy was down for several days, the scheduler automatically catches up
by exporting each missed day in order:

```
Miss 3 days (Feb 16, 17, 18=today):

  Next run (Feb 18):
    Export Feb 16 → marker = "2026-02-16"
    Export Feb 17 → marker = "2026-02-17"
    Feb 18 = today → skip

  Next run (Feb 19):
    Export Feb 18 → marker = "2026-02-18"
    Feb 19 = today → skip
```

The marker advances one day at a time, so if a single day's upload fails,
only that day is retried — earlier days are not re-exported.

### metricsMarker from Mavvrik

On `POST /mavvrik/init`, LiteLLM calls the Mavvrik register endpoint:
```
POST {api_endpoint}/{tenant}/k8s/agent/{instance_id}
Response: { "metricsMarker": <epoch_seconds> }
```

`metricsMarker` tells LiteLLM from what date Mavvrik wants data.
- If `metricsMarker > 0` → initial marker = that epoch converted to a date string
- If `metricsMarker == 0` or absent → initial marker = first day of current month

After each successfully exported day, LiteLLM also PATCHes Mavvrik:
```
PATCH {api_endpoint}/{tenant}/k8s/agent/{instance_id}
Body: { "metricsMarker": <epoch_of_exported_day> }
```
This keeps Mavvrik in sync with LiteLLM's export cursor.

---

## GCS Upload Pattern

The upload never touches GCP credentials directly. Mavvrik issues a
pre-signed GCS URL; LiteLLM uses it to upload via the GCS resumable upload protocol.

### Step 1 — Get signed URL
```
GET {api_endpoint}/{tenant}/k8s/agent/{instance_id}/upload-url
    ?name=2026-02-17&provider=k8s&type=metrics
Header: x-api-key: {api_key}
Response: { "url": "https://storage.googleapis.com/bucket/path?Signature=..." }
```

### Step 2 — Initiate resumable upload
```
POST {signed_url}
Headers: Content-Type: application/gzip, x-goog-resumable: start
Body: {"contentEncoding":"gzip","contentDisposition":"attachment"}
Response 201: Location: {session_uri}
```

### Step 3 — Upload data
```
PUT {session_uri}
Headers: Content-Type: application/gzip, Content-Encoding: gzip
Body: gzip(CSV bytes)
Response 200/201 → upload complete
```

### GCS object path
```
{bucket}/{tenant}/k8s/{instance_id}/metrics/{date}

Example:
  acme-bucket/my-tenant/k8s/litellm-prod/metrics/2026-02-17
```

**One file per calendar day.** Re-uploading the same date overwrites the existing
GCS object — exports are fully idempotent. No deduplication logic is required
on Mavvrik's side.

---

## Past Data (Backfill)

To re-export or backfill a specific date, use the manual export endpoint
with an explicit `date_str`. This does **not** affect the scheduled cursor.

```bash
# Export a single past day
curl -X POST http://localhost:4000/mavvrik/export \
  -H "Authorization: Bearer <master-key>" \
  -H "Content-Type: application/json" \
  -d '{"date_str": "2026-01-15"}'
```

To backfill a range of dates:
```bash
for date in 2026-01-01 2026-01-02 2026-01-03; do
  curl -X POST http://localhost:4000/mavvrik/export \
    -H "Authorization: Bearer <master-key>" \
    -H "Content-Type: application/json" \
    -d "{\"date_str\": \"${date}\"}"
  sleep 2
done
```

Re-uploading the same date is safe — it overwrites the GCS object with the
latest data. Manual exports do **not** advance the scheduled marker.

---

## Future Data (Scheduled Export)

The APScheduler job fires every `MAVVRIK_EXPORT_INTERVAL_MINUTES` (default: 60).
Each run exports all complete days since the last marker, up to yesterday.
It runs automatically as long as:
1. The `callbacks: ["mavvrik"]` line is in the proxy config YAML, AND
2. `mavvrik_settings` exists in `LiteLLM_Config` (i.e. `/mavvrik/init` has been called)

### Why today is never exported

`LiteLLM_DailyUserSpend` rows for today are still being updated every ~10–15
seconds as new API calls come in. Exporting a partial day would send incomplete
spend figures. The scheduler always stops at yesterday — the most recent day
where all rows are final and settled.

### Multi-replica safety

In Kubernetes with multiple proxy replicas, only one pod should export per run.
The integration uses LiteLLM's built-in **Pod Lock Manager** (Redis-backed):
- With Redis configured: only the pod that acquires the lock exports
- Without Redis: all pods export independently (same date, same GCS object —
  last writer wins, data is identical so this is safe)

### What happens if an export fails mid-loop

The marker advances one day at a time, **after** each successful upload:

```
Exporting Feb 16, Feb 17, Feb 18...
  Feb 16 uploaded → marker = "2026-02-16" ✓
  Feb 17 upload fails → marker stays "2026-02-16"

Next run:
  Retries from Feb 17 (marker + 1 day)
```

- Step 1 (signed URL) retries up to 3 times with exponential backoff (1s → 2s → 4s)
- 4xx errors fail immediately (retries won't fix auth or config issues)
- 5xx and network errors retry

---

## Data Deduplication

| Layer | Mechanism |
|-------|-----------|
| **LiteLLM side** | Marker advances per day — each day exported at most once per run |
| **GCS side** | Object named by date — re-upload overwrites, no duplicate objects |
| **Mavvrik side** | No deduplication needed — one file per day, last write wins |

---

## Marker Storage

The marker is stored as a date string inside the encrypted settings blob
in `LiteLLM_Config`:

```json
{
  "api_key":      "<encrypted>",
  "api_endpoint": "https://api.mavvrik.dev",
  "tenant":       "my-tenant",
  "instance_id":  "litellm-prod",
  "timezone":     "UTC",
  "marker":       "2026-02-17"
}
```

To inspect the current marker:
```bash
curl -H "Authorization: Bearer <master-key>" http://localhost:4000/mavvrik/settings
```

To reset the marker so the scheduler re-exports from a specific date:
```bash
curl -X PUT http://localhost:4000/mavvrik/settings \
  -H "Authorization: Bearer <master-key>" \
  -H "Content-Type: application/json" \
  -d '{"marker": "2026-01-31"}'
```

After this, the next scheduled run will export Feb 1 onwards.
