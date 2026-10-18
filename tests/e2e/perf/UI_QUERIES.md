# SpendLog UI query performance

Run `test_ui_queries.py` against a dedicated proxy and a disposable database restored from the metadata-only performance seed. These tests do not call an LLM or measure browser rendering

The suite measures 42 request shapes across 20 routes used by current dashboard screens. It replays direct HTTP requests against the existing bulk seed, with the seed anchor replacing the wall clock and a UTC browser timezone. It does not run a browser or claim to cover every dashboard request

## UI callers

All paths below are relative to `ui/litellm-dashboard/src/`. Each case requires an authenticated admin session with access to that screen, and selected resources must exist in the fixture

| Cases | UI action and source |
| --- | --- |
| `logs_*` | Logs > Request Logs, change date range, filter, sort or next page. `components/logs/request/useLogFilterLogic.ts` and `RequestLogsFilters.tsx` |
| `usage_*` | Usage overview or entity tab, open key activity, search keys, expand a model group. `app/(dashboard)/usage/_components/components/UsagePageView.tsx`, `EntityUsage/EntityUsage.tsx`, `EntityUsage/entityFetchFns.ts`, `components/UsagePage/components/KeyActivityPanel.tsx` |
| `team_users_*` | Team Usage. `EntityUsage/EntityUsage.tsx` passes selected teams or all available non-dashboard teams to `TeamUserSpendCard.tsx` |
| `cache_activity_*` | Caching > Cache Analytics, optionally filter aliases/models. `app/(dashboard)/hooks/caching/useCacheActivity.ts` |
| `cache_leakage_keys` | Cost Optimization, switch cache leakage to Keys. `app/(dashboard)/cost-optimization/_components/useCacheLeakageKeys.ts` |

Request serialization lives in `components/networking.tsx`. Logs use page size 25 and session grouping, including the next-session cursor returned by page one. Individual Logs filters use a 30-day window to exercise larger matching datasets. Their case names end in `_30d` so the measurements do not share history with the old 24-hour filters. The synthetic cursorless page-three request is excluded

Usage list, search and model drilldown requests do not inject an `api_key` filter. Team activity excludes `litellm-dashboard`. Cache leakage supplies `include_current_utc_day=true`. Dates and filter values are deterministic fixture values, not captured production traffic

Old Usage is excluded because its expensive-query gate normally hides those calls at this database size. LiteAdmin's agent-selected reports are excluded from this deterministic dashboard set. Missing-token alias recovery is backend functional coverage, not a separate UI action, and its artificial requests and orphan fixtures are removed

This mapping is source-verified, not browser-recorded. Recheck the named callers when changing cases. Neither browser capture nor an automatic UI-drift checker is installed by this suite

## Dataset scope

Logs, team-user spend and cache activity query the bulk SpendLogs data. User, team, tag and customer usage, plus cache leakage, query daily tables aggregated from that dataset. These are sequential large-dataset queries, not concurrent traffic or saturation tests

A read-only restore of `db-perf-v1-105-1tb` gave exact request totals from daily aggregates: 9,442,606 across 24 hours, 66,077,435 across seven days and 283,189,684 across 30 days. The retained 30-day key filter matches 1,134,559 requests, user filter 11,316,454, team filter 113,298,119 and model filter 94,395,473. These describe matching requests, not rows scanned or returned by the API, and are documentation rather than performance assertions

PostgreSQL table-wide estimates (`pg_class.reltuples`) are approximately 2.70 million daily user rows, 2.70 million daily team rows, 5.40 million daily tag rows and 78.27 million daily end-user rows. These are not exact counts within each Usage case's seven-day window

The individual end-user filter is excluded because it matched only 104 requests in 24 hours and 8,616 across all dates. User/end-user dropdown requests are also excluded because their queries cap the input at 10,000 matching SpendLogs before extracting distinct values. Neither is a broad matching-data workload in this seed

Excluded because they only had tiny matching fixtures or individual-record lookups: organization/agent usage, guardrail/tool logs, request-ID lookup, payload/session drawers, session/search filters for `perf-ui-*`, the OAuth-token filter, and all prompt-caching request cases. The seed has response-cache hits but no prompt-cache usage or injection metadata, so the prompt-caching cases only matched the 320 added rows. No UI-specific rows are inserted anymore

| Remaining cases | Count | Variations |
| --- | ---: | --- |
| Logs | 21 | 24h/7d/30d, page 2 with session cursor, 12 individual filters over 30 days, five additional descending sorts |
| User/team/tag/customer usage | 16 | Aggregated totals, keys, key search, model top keys for each |
| Team-user spend | 2 | All 100 seeded teams or `perf-team-0` |
| Cache activity | 2 | All or selected aliases/model |
| Cache leakage keys | 1 | 30-day window |

## Deployment prerequisites

Use a fresh proxy against the disposable restore, never the active seed source or a shared test database. Apply the target migrations and wait for readiness first. The source seed's `litellm_perf_seed.checkpoint` must exist and contain rows

Configure `litellm_settings.service_callback: [prometheus_system]` and expose metrics for every proxy worker/pod. Set `E2E_DATABASE_METRICS_URLS` to their comma-separated URLs, not a load-balanced endpoint that can miss a pod. Absent failure counters count as zero only when the endpoint exposes a positive PostgreSQL request counter. Missing instrumentation stops the run

No expensive-query override or legacy view creation is needed for these cases. Keep the normal large-database UI restrictions

The cases use explicit dates from the seed anchor where the UI supplies dates. A missing nonempty fixture result records a contract failure rather than reporting a fast empty query as success

## Run

Set `DATABASE_URL` and `LITELLM_MASTER_KEY` securely for this disposable deployment. The URL must name the same database used by the proxy, with hostname `db-perf-perf-<run>.litellm.svc`. The explicit disposable confirmation permits cancellation of leftover queries in this database. Setup reads the existing seed checkpoint without adding UI-specific rows. Destroy the disposable database after the run

```bash
export LITELLM_PROXY_URL=http://localhost:4000
export E2E_DATABASE_METRICS_URLS=http://localhost:4000/metrics
export E2E_PERF_TESTS=1
export E2E_UI_PERF_DISPOSABLE=1
export E2E_RESET_SPEND_LOGS=0
export E2E_ARTIFACT_DIR="$PWD/ui-query-results"
mkdir -p "$E2E_ARTIFACT_DIR"
uv run pytest tests/e2e/perf/test_ui_queries.py --reruns 0 --junitxml="$E2E_ARTIFACT_DIR/junit.xml" -q
```

Do not enable SpendLog retention or run other traffic on this stack. The `quiet_stack` marker coordinates local pytest callers but cannot stop external traffic. Use one pytest worker for comparable timings

`E2E_UI_QUERY_REPEATS` defaults to 9, giving ten measured requests including the first, and `E2E_UI_QUERY_TIMEOUT_SECONDS` defaults to 120. Every request is issued once without transport retries. A failed first request stops that case's repeated samples. Cursor prerequisites are recorded separately, and failed prerequisites stop that case's measurements

## Reading results

Each case writes `ui-query-<name>.json` with its exact method, path, parameters and body, response status counts, elapsed times, row counts, errors and before/after PostgreSQL failure counters. Response bodies and authentication headers are not saved

The dashboard's `perf_ui_seconds_p50`, `_p90` and `_max` include all successful measured requests, including the first. `perf_ui_samples` is their sample count. p50 is the median and p90 uses nearest rank, the ninth sorted value with ten successes. Failures and cursor prerequisites contribute no latency. Pass percentage is `(perf_ui_requests - perf_ui_failures) / perf_ui_requests * 100`, including cursor prerequisites in the request counts

The first-request and repeated-request metrics are retained for existing consumers. This is not a cold-cache claim: previous cases and previous runs can warm database buffers or application caches. Restart the disposable proxy between independent runs when comparing independent runs

Repeated p50 is reported for successful samples, p95 only with at least 20 and p99 only with at least 100. First-request latency is omitted when that request fails. HTTP, JSON, response-shape, nonempty-result, cursor failures are recorded, never asserted. Endpoint failures, database counter increases and latency do not fail CI

Each JUnit testcase `test_ui_query_performance[<case>]` carries `perf_ui_requests`, `perf_ui_failures`, `perf_ui_repeated_samples`, and nonzero `perf_ui_error_<code>` counts. Codes distinguish `http_400`, `http_500`, `http_503`, other HTTP statuses, `timeout`, `network`, `invalid_json` and `contract`. Successful latency properties are `perf_ui_first_seconds` and `perf_ui_repeated_seconds_p50`, `_p95`, `_p99`, `_max`. A failed cursor prerequisite is counted once and contributes no latency

Before every case the runner requires three idle database observations and stable failure counters. After each case it cancels leftover SQL, terminates idle transactions and waits for idle and stable counters again. `ui-isolation-<case>.jsonl` retains active SQL, waits, elapsed times, raw samples and counter observations. `perf_ui_database_failures` measures the increase before cleanup, while `perf_ui_cleanup_database_failures` measures the increase observed afterward, including cancellations and potentially delayed request failures. Isolation, instrumentation or setup failures still stop the run rather than publish contaminated measurements. Disk and process caches are not reset between cases

The Buildkite `ui_queries` lane restores the pinned snapshot independently, waits for migrations and indexes to complete, runs these cases with no retries, uploads JUnit and JSON artifacts, and removes the temporary resources. It follows the nightly regular perf and database-upgrade builds using the same LiteLLM SHA. Shinkiro ingestion is a separate change

Validate a small restore first. Small-database numbers validate the runner only, not full-size performance. Run the same cases against the completed full-size snapshot before publishing large-database results
