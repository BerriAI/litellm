# Guardrails Usage Dashboard - Implementation Status

## Completed: Backend Implementation (Phases 1-4)

### ✅ Phase 1: Database Schema
- **File**: `litellm/proxy/schema.prisma`
- Added `LiteLLM_DailyGuardrailMetrics` table with:
  - Unique constraint on `[guardrail_name, guardrail_provider, guardrail_mode, date, api_key]`
  - Indexes on `date`, `guardrail_name`, `guardrail_provider`, `api_key`
  - Fields for tracking total_requests, success/intervened/failed/not_run counts
  - Aggregated latency metrics in milliseconds
- **Action Required**: Run `poetry run prisma migrate dev --name add_guardrail_metrics` to create the table

### ✅ Phase 2: Data Collection & Aggregation
- **File**: `litellm/proxy/_types.py`
  - Added `DailyGuardrailMetricsTransaction` TypedDict

- **File**: `litellm/proxy/db/db_spend_update_writer.py`
  - Added `daily_guardrail_metrics_update_queue` to `__init__`
  - Implemented `add_spend_log_transaction_to_daily_guardrail_transaction()`:
    - Extracts guardrail_information from metadata
    - Creates separate transaction per guardrail
    - Calculates status counts and latency in milliseconds
  - Implemented `update_daily_guardrail_metrics()` static method:
    - Batch upserts to database with retry logic
    - Increments counters on conflict
  - Added guardrail transaction call to `update_database()` flow
  - Added commit logic to `_commit_spend_updates_to_db_without_redis_buffer()`

### ✅ Phase 3: Type Definitions
- **File**: `litellm/types/proxy/management_endpoints/guardrail_metrics.py`
  - Created Pydantic models for:
    - `GuardrailMetrics` - Aggregated metrics
    - `GuardrailSummary` - Table view
    - `GuardrailMetricsResponse` - List endpoint response
    - `GuardrailDailyMetrics` - Time-series data
    - `GuardrailDetailMetrics` - Detail view with daily metrics
    - `GuardrailLogEntry` - Individual request log
    - `GuardrailLogsResponse` - Logs endpoint response

### ✅ Phase 4: API Endpoints
- **File**: `litellm/proxy/management_endpoints/guardrail_metrics_endpoints.py`
  - Implemented 3 endpoints:
    1. `GET /guardrail/metrics` - List guardrails with aggregated metrics
       - Query params: start_date, end_date, guardrail_name, provider, page, page_size
       - Returns sorted by fail_rate descending
    2. `GET /guardrail/{guardrail_name}/metrics` - Detail view metrics
       - Returns overview stats + daily time-series
    3. `GET /guardrail/{guardrail_name}/logs` - Request logs
       - Query params: start_date, end_date, status_filter, page, page_size
       - Filters LiteLLM_SpendLogs by guardrail_information

- **File**: `litellm/proxy/proxy_server.py`
  - Added import for `guardrail_metrics_router`
  - Registered router with `app.include_router(guardrail_metrics_router)`

## 🔲 Pending: Frontend Implementation (Phase 5)

### TypeScript Types
- **File to create**: `ui/litellm-dashboard/src/components/GuardrailsPage/types.ts`
- Defines interfaces matching backend Pydantic models

### Networking Functions
- **File to update**: `ui/litellm-dashboard/src/components/networking.tsx`
- Add API call functions:
  - `guardrailMetricsCall()`
  - `guardrailDetailMetricsCall()`
  - `guardrailLogsCall()`

### Components
1. **Table View**: `ui/litellm-dashboard/src/components/GuardrailsPage/GuardrailsTableView.tsx`
   - Displays list of guardrails with metrics
   - Clickable rows navigate to detail view

2. **Detail View**: `ui/litellm-dashboard/src/components/GuardrailsPage/GuardrailDetailView.tsx`
   - Metric cards (Requests, Fail Rate, Latency, Blocked)
   - Tabs for Overview and Logs
   - Area chart for fail rate trend

3. **Logs Tab**: `ui/litellm-dashboard/src/components/GuardrailsPage/GuardrailLogsTab.tsx`
   - Filterable table (All, Blocked, Passed)
   - Expandable rows for guardrail response details

### Pages
1. **Main Page**: `ui/litellm-dashboard/src/app/(dashboard)/guardrails/page.tsx`
   - Date range picker
   - Renders GuardrailsTableView

2. **Detail Page**: `ui/litellm-dashboard/src/app/(dashboard)/guardrails/[name]/page.tsx`
   - Back button to overview
   - Date range picker
   - Renders GuardrailDetailView

## Testing & Verification

### Backend Testing Steps
1. Run Prisma migration:
   ```bash
   cd /Users/krrishdholakia/Documents/litellm-guardrails-dashboard
   poetry run prisma migrate dev --name add_guardrail_metrics --schema=litellm/proxy/schema.prisma
   ```

2. Start the proxy server with guardrails configured

3. Send requests that trigger guardrails (both pass and fail cases)

4. Wait 60s for batch commit to database

5. Test API endpoints:
   ```bash
   # List guardrails
   curl -X GET "http://localhost:4000/guardrail/metrics?start_date=2026-02-01&end_date=2026-02-19" \
     -H "Authorization: Bearer <token>"

   # Get guardrail details
   curl -X GET "http://localhost:4000/guardrail/my-guardrail/metrics?start_date=2026-02-01&end_date=2026-02-19" \
     -H "Authorization: Bearer <token>"

   # Get logs
   curl -X GET "http://localhost:4000/guardrail/my-guardrail/logs?start_date=2026-02-01&end_date=2026-02-19" \
     -H "Authorization: Bearer <token>"
   ```

6. Verify data in database:
   ```sql
   SELECT * FROM "LiteLLM_DailyGuardrailMetrics" LIMIT 10;
   ```

### Frontend Testing Steps (Once Implemented)
1. Navigate to `/guardrails` page
2. Verify table loads with metrics
3. Click guardrail row → navigate to detail page
4. Verify Overview tab shows metrics and chart
5. Switch to Logs tab → verify logs display
6. Test status filter (All, Blocked, Passed)
7. Test date range filtering
8. Test pagination

## Key Design Decisions

### Fail Rate Calculation
- **Formula**: `(intervened_count / total_requests) * 100`
- Only counts `guardrail_intervened` as failures (policy violations)
- Excludes `guardrail_failed_to_respond` (infrastructure errors)

### Average Latency
- **Formula**: `total_latency_ms / total_requests`
- Measures **guardrail execution overhead** (not total request latency)
- Captured from `StandardLoggingGuardrailInformation.duration` field
- Stored in milliseconds for better precision

### Per-Request Logs
- No new table needed
- Existing `LiteLLM_SpendLogs.metadata.guardrail_information` used
- Query optimized with date filtering and over-fetching strategy

### Aggregation Strategy
- Daily aggregation reduces full table scans on large datasets
- Batch commits every 60s reduce database load
- Indexed queries for fast filtering
- Pagination limits memory usage

## Files Modified

### Backend
1. `litellm/proxy/schema.prisma` - Added table
2. `litellm/proxy/_types.py` - Added transaction type
3. `litellm/proxy/db/db_spend_update_writer.py` - Data collection & commit
4. `litellm/types/proxy/management_endpoints/guardrail_metrics.py` - New file
5. `litellm/proxy/management_endpoints/guardrail_metrics_endpoints.py` - New file
6. `litellm/proxy/proxy_server.py` - Router registration

### Frontend (Pending - 7 files)
1. `ui/litellm-dashboard/src/components/GuardrailsPage/types.ts`
2. `ui/litellm-dashboard/src/components/networking.tsx`
3. `ui/litellm-dashboard/src/components/GuardrailsPage/GuardrailsTableView.tsx`
4. `ui/litellm-dashboard/src/components/GuardrailsPage/GuardrailDetailView.tsx`
5. `ui/litellm-dashboard/src/components/GuardrailsPage/GuardrailLogsTab.tsx`
6. `ui/litellm-dashboard/src/app/(dashboard)/guardrails/page.tsx`
7. `ui/litellm-dashboard/src/app/(dashboard)/guardrails/[name]/page.tsx`

## Next Steps

1. **Run Prisma Migration** to create the database table
2. **Test Backend** with curl requests after sending guardrail traffic
3. **Implement Frontend** following Phase 5 specifications
4. **Run `make test-unit`** to ensure no regressions
5. **Test Integration** with high request volume (1000+ requests)
6. **Create Pull Request** with proper tests and documentation

## Notes

- Migration needs to be run in an environment with proper Prisma setup
- Some Pyright diagnostics appeared but are pre-existing in codebase
- Frontend implementation follows existing LiteLLM dashboard patterns (Tremor UI components)
- All backend code follows existing patterns from daily spend tracking
