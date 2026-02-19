# Mavvrik Integration — Architecture & Design

**Audience:** Engineering architects, platform teams, integration reviewers

**Purpose:** Explain how LiteLLM exports usage data to Mavvrik for cost tracking and analysis

---

## Executive Summary

LiteLLM exports daily aggregated usage data to Mavvrik's cost tracking platform. The integration runs automatically in the LiteLLM proxy, uploading one compressed CSV file per calendar day to Google Cloud Storage via Mavvrik's API.

**Key characteristics:**
- One file per day (idempotent, overwrites on re-upload)
- Automatic backfill on first run
- Cursor synchronization with Mavvrik to honor resets
- Multi-replica safe (Redis-based locking)
- No customer credentials touch GCS directly

---

## Architecture Overview

### Components

**LiteLLM Proxy (FastAPI)**
- Admin API endpoints for configuration and manual triggers
- APScheduler background job (runs every 60 minutes)
- Integration core classes (logger, database, transformer, streamer)
- Pod Lock Manager (Redis) prevents duplicate exports in multi-pod deployments

**PostgreSQL Database**
- `LiteLLM_DailyUserSpend`: aggregated daily usage (one row per user/date/key/model/provider)
- `LiteLLM_Config`: encrypted settings and export cursor (marker)
- Supporting tables: teams, users, API keys for data enrichment

**Mavvrik API**
- Register endpoint: verifies connectivity, returns metricsMarker cursor
- Upload URL endpoint: issues pre-signed GCS URLs
- Advance endpoint: updates Mavvrik's cursor after each successful export

**Google Cloud Storage**
- Final destination for CSV files
- Object path: `{bucket}/{tenant}/k8s/{instance_id}/metrics/{YYYY-MM-DD}`
- Resumable upload protocol (3-step: initiate → upload → finalize)

### Data Flow

```
API Calls → LiteLLM_DailyUserSpend (upserted every ~15 sec)
                ↓
Scheduler (hourly) → Query yesterday's data
                ↓
Transform to CSV → Compress with gzip
                ↓
Get signed URL from Mavvrik → Upload to GCS
                ↓
Advance marker locally → Sync cursor with Mavvrik
```

---

## Export Mechanism

### The Date-Based Cursor

The integration exports one calendar day at a time. A **marker** (YYYY-MM-DD string) tracks the last successfully exported date. The scheduler exports each missed day from `marker + 1` up to yesterday.

**Why yesterday, not today?**

The source table receives new rows every 10–15 seconds as API calls complete. Exporting today would send incomplete data. Yesterday is the most recent day where all rows are final.

**Cursor advancement:**

After each successful upload, the marker advances to that date. If a day fails, the marker stays at the previous date, so the next run retries from the failed day forward.

### First Run Behavior

On the first export, no marker exists. The scheduler queries `MIN(date)` from the source table and uses that as the starting point, automatically backfilling all historical data without manual intervention.

**Fallback:** If the table is empty or the query fails, the scheduler starts from yesterday.

### Cursor Synchronization with Mavvrik

**On every scheduled run**, the integration calls Mavvrik's register endpoint to:
1. Verify connectivity
2. Retrieve Mavvrik's `metricsMarker` (epoch timestamp)

**If Mavvrik's cursor is earlier than the local cursor**, the local marker moves back to honor Mavvrik's reset. This allows Mavvrik to request re-export by resetting its cursor.

**If the register call fails**, the run continues with the local marker (best-effort, non-fatal).

**After each successful export**, the integration PATCHes Mavvrik's `metricsMarker` to keep both sides in sync.

### Idempotency

Re-uploading the same date is safe. The GCS object name equals the date (YYYY-MM-DD), so re-upload overwrites the existing file. No deduplication logic is required downstream.

---

## Source Data

### LiteLLM_DailyUserSpend Table

**Schema characteristics:**
- One row per `(user_id, date, api_key, model, provider)` combination
- `date` column: TEXT type (YYYY-MM-DD format, not PostgreSQL DATE)
- Rows accumulate throughout the day via Prisma upsert
- Token counts and spend values are summed, not duplicated

**Write pattern:**
- Every API call enqueues spend data in memory
- Every ~10–15 seconds, a batch flush upserts rows to the database
- `updated_at` reflects the last batch flush, not individual API calls

**Export query:**
```sql
SELECT dus.*, vt.team_id, vt.key_alias, tt.team_alias, ut.user_email
FROM "LiteLLM_DailyUserSpend" dus
LEFT JOIN "LiteLLM_VerificationToken" vt ON dus.api_key = vt.token
LEFT JOIN "LiteLLM_TeamTable" tt ON vt.team_id = tt.team_id
LEFT JOIN "LiteLLM_UserTable" ut ON dus.user_id = ut.user_id
WHERE dus.date = $1
```

**Exported columns (19 total):**
- Usage metrics: tokens (prompt, completion, cache), requests (total, successful, failed), spend
- Identifiers: user ID, hashed API key, model, provider
- Enrichments: team ID/alias, API key alias, user email
- Timestamps: created_at, updated_at

---

## Transformation & Upload

### CSV Format

The transformer converts the Polars DataFrame to CSV with headers. The first row contains column names; subsequent rows contain data.

**Compression:** gzip applied before upload reduces bandwidth and storage cost.

**Encoding:** UTF-8 throughout (query → DataFrame → CSV → gzip → GCS).

### GCS Resumable Upload Protocol

LiteLLM uses the 3-step resumable upload pattern:

**Step 1: Get signed URL**
```
GET {mavvrik_api}/{tenant}/k8s/agent/{instance_id}/upload-url?name=YYYY-MM-DD
Response: { "url": "https://storage.googleapis.com/..." }
```

**Step 2: Initiate resumable session**
```
POST {signed_url}
Headers: x-goog-resumable: start, Content-Type: application/gzip
Response 201: Location: {session_uri}
```

**Step 3: Upload data**
```
PUT {session_uri}
Headers: Content-Type: application/gzip, Content-Encoding: gzip
Body: gzip(CSV bytes)
Response 200/201: upload complete
```

**Retry logic:**
- 3 attempts with exponential backoff (1s → 2s → 4s)
- 4xx errors fail immediately (auth/config issues)
- 5xx and network errors retry

---

## Deployment Architecture

### Scheduler Configuration

The APScheduler job fires every `MAVVRIK_EXPORT_INTERVAL_MINUTES` (default: 60). Each run:
1. Acquires Redis lock (if Redis configured)
2. Calls Mavvrik register endpoint (verify connectivity, get cursor)
3. Exports each missed day from effective marker to yesterday
4. Releases lock

**Environment variables:**
- `MAVVRIK_TENANT_ID`: Mavvrik tenant identifier
- `MAVVRIK_INSTANCE_ID`: LiteLLM instance identifier
- `MAVVRIK_ENDPOINT`: Mavvrik API base URL
- `MAVVRIK_API_KEY`: API key for Mavvrik authentication
- `MAVVRIK_EXPORT_INTERVAL_MINUTES`: scheduler frequency (default: 60)

### Multi-Replica Safety

In Kubernetes deployments with multiple proxy replicas:

**With Redis configured:**
- One pod acquires the lock per scheduled run
- Other pods skip the run

**Without Redis:**
- All pods export independently
- Each pod uploads the same date to the same GCS object
- Last writer wins (data is identical, safe)

The Redis lock prevents duplicate API calls to Mavvrik and duplicate GCS uploads but does not affect correctness.

### Activation Requirements

The scheduler starts automatically when **both** conditions are met:
1. `success_callback: ["mavvrik"]` or `callbacks: ["mavvrik"]` present in proxy config YAML
2. Settings exist in `LiteLLM_Config` (via `/mavvrik/init` endpoint)

**Callback field options:**
- `success_callback: ["mavvrik"]` — logs successful API calls only (recommended for cost tracking)
- `callbacks: ["mavvrik"]` — logs both successful and failed calls
- `failure_callback: ["mavvrik"]` — logs failed calls only (not useful for cost data)

If either condition is missing, the integration remains dormant.

---

## API Reference

### Admin Endpoints

**POST /mavvrik/init**
- Initialize Mavvrik settings (one-time setup)
- Calls Mavvrik register endpoint to verify credentials
- Encrypts and stores settings in database
- Returns initial cursor from Mavvrik's metricsMarker

**GET /mavvrik/settings**
- Retrieve current settings and marker
- Returns decrypted settings (API key partially masked)

**PUT /mavvrik/settings**
- Update marker or other settings
- Use to reset cursor for re-export

**POST /mavvrik/dry-run**
- Preview CSV output for a specific date
- Does not upload or advance cursor

**POST /mavvrik/export**
- Manual export for a specific date
- Does not affect scheduled cursor
- Use for backfill or testing

All endpoints require master key authentication.

---

## Error Handling & Observability

### Logging

The integration logs at INFO level for normal operations:
- Scheduler start/stop
- Register calls and cursor synchronization
- Each date exported successfully
- Marker advancement

Warnings logged for:
- Register endpoint failures (non-fatal)
- Partial export failures (cursor not advanced)

Errors logged for:
- Database query failures
- GCS upload failures (after retries exhausted)
- Invalid configuration

### Failure Recovery

**Database query fails:** Run skips, logs error, retries next hour

**GCS upload fails (single date):** Marker stays at previous date, retries that date on next run

**Register call fails:** Warning logged, run continues with local marker

**Redis lock acquisition fails:** Pod skips run (other pod handles export)

### Metrics

LiteLLM's built-in Prometheus integration tracks:
- `litellm_callbacks_total{callback="mavvrik"}`: successful callbacks
- `litellm_callbacks_failed{callback="mavvrik"}`: failed callbacks

The scheduler does not increment these counters directly (runs on background thread), but integration errors surface in proxy logs.

---

## Security Model

### Credential Management

**Mavvrik API key:** Stored encrypted in `LiteLLM_Config`, decrypted only for API calls

**GCS access:** LiteLLM never sees GCS credentials; Mavvrik issues pre-signed URLs with embedded auth

**Database encryption:** Settings blob encrypted at rest using LiteLLM's standard encryption

**Transit security:** All API calls use HTTPS

### Data Exposure

**Personal data in exports:**
- User email (if configured in LiteLLM)
- Team names and API key aliases
- User IDs (LiteLLM internal identifiers)

**Sensitive data NOT exported:**
- Raw API keys (only hashed keys exported)
- Prompt content or completion text
- Customer IP addresses

---

## Testing Strategy

### Unit Tests (40 tests)

**File:** `tests/test_litellm/integrations/mavvrik/test_mavvrik_stream_api.py`

**Coverage:**
- Mavvrik API calls with mocked HTTP responses
- Error handling and retry logic
- Cursor advancement and synchronization

### E2E Tests (9 tests)

**File:** `tests/test_litellm/integrations/mavvrik/test_e2e_mavvrik_stream_api.py`

**Coverage:**
- Real Mavvrik API and GCS integration
- Register → upload → advance cycle
- Idempotent re-upload

**Prerequisites:** Valid Mavvrik credentials in environment

### Integration Tests

**File:** `tests/test_litellm/integrations/mavvrik/test_transform.py`

**Coverage:**
- CSV transformation
- Column ordering and formatting
- gzip compression

---

## Performance Characteristics

### Data Volume

**Typical export size per day:**
- 1,000 API calls/day ≈ 50 KB compressed CSV
- 10,000 API calls/day ≈ 500 KB compressed CSV
- 100,000 API calls/day ≈ 5 MB compressed CSV

**Memory usage:**
- Query result loaded into Polars DataFrame (resident in memory)
- CSV string held in memory before compression
- gzip compression operates in memory
- Peak memory ≈ 3× uncompressed CSV size

### Network Usage

**Per scheduled run (assuming 1 missed day):**
- 1 register call (< 1 KB)
- 1 signed URL request (< 1 KB)
- 1 GCS upload (compressed CSV size)
- 1 advance call (< 1 KB)

**Total:** Compressed CSV size + negligible overhead

### Database Load

**Per scheduled run:**
- 1 query for settings
- 1 query for usage data (filtered by date)
- 1 update to advance marker

**Query complexity:** O(n) where n = rows for that date (typically 100–10,000)

**Index requirement:** Index on `LiteLLM_DailyUserSpend.date` for fast filtering

---

## Operational Considerations

### Backfill Procedure

**Re-export a single date:**
```bash
curl -X POST http://localhost:4000/mavvrik/export \
  -H "Authorization: Bearer <master-key>" \
  -H "Content-Type: application/json" \
  -d '{"date_str": "2026-01-15"}'
```

**Re-export a range:**
```bash
for date in $(seq -f "2026-01-%02g" 1 31); do
  curl -X POST http://localhost:4000/mavvrik/export \
    -H "Authorization: Bearer <master-key>" \
    -H "Content-Type: application/json" \
    -d "{\"date_str\": \"${date}\"}"
  sleep 2
done
```

Manual exports do not affect the scheduled marker.

### Cursor Reset

**To re-export from a specific date going forward:**
```bash
curl -X PUT http://localhost:4000/mavvrik/settings \
  -H "Authorization: Bearer <master-key>" \
  -H "Content-Type: application/json" \
  -d '{"marker": "2026-01-31"}'
```

Next scheduled run exports Feb 1 onwards.

**Mavvrik-initiated reset:** Mavvrik can reset its `metricsMarker` cursor. LiteLLM honors the reset on the next scheduled run (register call detects earlier cursor, moves local marker back).

### Monitoring

**Check scheduler status:**
```bash
curl http://localhost:4000/health/readiness | jq .mavvrik_scheduler_running
```

**Check current marker:**
```bash
curl -H "Authorization: Bearer <master-key>" http://localhost:4000/mavvrik/settings | jq .marker
```

**Verify exports in GCS:** Check Mavvrik UI or GCS console for files at `{bucket}/{tenant}/k8s/{instance_id}/metrics/`

---

## Design Decisions

### Why Date-Based, Not Time-Based?

**Decision:** Export one file per calendar day (YYYY-MM-DD), not hourly or minute-based intervals.

**Rationale:**
- Source table aggregates by day (natural grain)
- GCS object names map cleanly to dates (idempotent overwrites)
- Downstream consumers group by day for cost analysis
- Simplifies cursor logic (one marker per day vs. complex interval tracking)

### Why Cursor in LiteLLM, Not Just Mavvrik?

**Decision:** Store marker locally in `LiteLLM_Config`, sync with Mavvrik's `metricsMarker`.

**Rationale:**
- LiteLLM continues exporting if Mavvrik's register call fails (resilient)
- Local cursor allows manual reset via LiteLLM admin API
- Mavvrik cursor enables Mavvrik-initiated re-export requests
- Dual cursor with sync on each run balances control and resilience

### Why Call Register on Every Run?

**Decision:** POST to Mavvrik's register endpoint at the start of every scheduled export.

**Rationale:**
- Verifies Mavvrik API is reachable before attempting upload
- Detects Mavvrik-initiated cursor resets (honor earlier marker)
- Fails fast if credentials are invalid or Mavvrik is down
- Non-fatal: if register fails, run continues with local marker (best-effort)

### Why Never Export Today?

**Decision:** Scheduler always stops at yesterday, never exports today's data.

**Rationale:**
- Today's rows are still accumulating (upserted every ~15 seconds)
- Exporting partial day sends incomplete spend figures
- Downstream dashboards expect complete daily totals
- Yesterday is the most recent day where all rows are final

---

## Future Enhancements

**Potential improvements not currently implemented:**

1. **Incremental updates within a day:** Export today's partial data for real-time dashboards (requires new endpoint or parameter)
2. **Compression format options:** Support Parquet or Avro for larger exports (CSV is simple, widely compatible)
3. **Delta exports:** Only export new/changed rows instead of full day (requires change tracking in source table)
4. **Multi-tenant isolation:** Support multiple Mavvrik tenants in one LiteLLM proxy (requires tenant routing)
5. **Export notifications:** Webhook or callback when each export completes (for monitoring)

---

## References

**Code files:**
- `litellm/integrations/mavvrik/mavvrik.py` — scheduler entry point
- `litellm/integrations/mavvrik/database.py` — database queries
- `litellm/integrations/mavvrik/transform.py` — CSV transformation
- `litellm/integrations/mavvrik/mavvrik_stream_api.py` — Mavvrik API client
- `litellm/proxy/spend_tracking/mavvrik_endpoints.py` — FastAPI admin endpoints

**Documentation:**
- `docs/mavvrik-architecture.md` — Mermaid diagrams
- `docs/mavvrik-data-flow.md` — detailed data flow reference
- `docs/mavvrik-customer-onboarding.md` — customer setup guide

**Tests:**
- `tests/test_litellm/integrations/mavvrik/` — unit and E2E tests

---

## Conclusion

The Mavvrik integration exports LiteLLM's daily usage data to Google Cloud Storage via Mavvrik's API. The design prioritizes simplicity (one file per day), resilience (cursor synchronization, retry logic), and operational safety (idempotent uploads, multi-replica locking). The integration runs automatically after one-time setup, requiring no ongoing manual intervention.
