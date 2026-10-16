# SpendLog UI query performance

Run `test_ui_queries.py` against a dedicated proxy and a disposable database restored from the metadata-only performance seed. These tests do not call an LLM or measure browser rendering

The suite measures 87 request shapes across 48 routes. Cases follow callers in `ui/litellm-dashboard/src/components/networking.tsx` and the mounted Logs, Usage, Old Usage, Caching, Prompt Caching, Guardrails, Tool Policies and LiteAdmin components

| Artifact surface | Requests |
| --- | --- |
| `current` | Logs list, request detail, session, user/end-user facets, team usage by user, cache activity, prompt-caching requests, guardrail logs and tool logs |
| `old_usage` | Global activity/model activity, provider spend, tags/tag names, logs, keys, teams, models and end-user spend |
| `liteadmin` | Agent-selected global, key and team spend reports, not automatic page-load traffic |
| `key_recovery` | Aggregated, keys, search and model-top-keys routes for user, team, tag, organization, customer and agent, plus user cache-leakage keys |

The ordinary daily-activity queries use daily tables. The `key_recovery` fixture deliberately omits token metadata and verifies the alias recovered from SpendLogs. Daily exports and the user pagination route do not perform this recovery and are excluded. Historical model-metrics routes with no mounted UI caller are also excluded

Logs cover the default 24-hour window, 7/30-day windows, UI filters/sorts, offset pagination and session cursors. Prompt-caching pagination uses the returned timestamp and request ID. Cursor page two must not repeat page-one IDs

## Deployment prerequisites

Use a fresh proxy against the disposable restore, never the active seed source or a shared test database. Apply the target migrations and wait for readiness first. The source seed's `litellm_perf_seed.checkpoint` must exist and contain rows

Configure `litellm_settings.service_callback: [prometheus_system]` and expose metrics for every proxy worker/pod. Set `E2E_DATABASE_METRICS_URLS` to their comma-separated URLs, not a load-balanced endpoint that can miss a pod. Both PostgreSQL and batch-write failure counter families must be present, and all counters must remain zero. Counters that were already nonzero fail the run too

The old-usage routes require the views from `db_scripts/create_views.py`. Create missing views on the disposable database before starting the proxy, not on the baseline. Enable expensive queries on this dedicated proxy with `DISABLE_EXPENSIVE_DB_QUERIES=false` and a `MAX_SPENDLOG_ROWS_TO_QUERY` above the seed's row count. These routes are normally hidden by the UI at this scale

The cases use explicit dates from the seed anchor where the UI supplies dates. Some legacy views always use the database clock's last 30 days. An old snapshot with no matching rows intentionally fails, rather than reporting fast empty queries as success. Refresh that fixture explicitly instead of rewriting the baseline timestamps

## Run

Set `DATABASE_URL` and `LITELLM_MASTER_KEY` securely for this disposable deployment. The URL must name the same database used by the proxy. The explicit disposable confirmation permits bounded fixture inserts, including 320 specialized logs, 25 orphan-key logs, daily rows and guardrail/tool index rows. Destroy the disposable database after the run

```bash
export LITELLM_PROXY_URL=http://localhost:4000
export E2E_DATABASE_METRICS_URLS=http://localhost:4000/metrics
export E2E_PERF_TESTS=1
export E2E_UI_PERF_DISPOSABLE=1
export E2E_RESET_SPEND_LOGS=0
export E2E_ARTIFACT_DIR="$PWD/ui-query-results"
mkdir -p "$E2E_ARTIFACT_DIR"
uv run pytest tests/e2e/perf/test_ui_queries.py --reruns 0 -q
```

Do not enable SpendLog retention or run other traffic on this stack. The `quiet_stack` marker coordinates local pytest callers but cannot stop external traffic. Use one pytest worker for comparable timings

`E2E_UI_QUERY_REPEATS` defaults to 5 and `E2E_UI_QUERY_TIMEOUT_SECONDS` to 120. Every request is issued once without transport retries. A failed first request stops that case's repeated samples. Cursor prerequisites are recorded separately, and failed prerequisites fail the case

## Reading results

Each case writes `ui-query-<name>.json` with its exact method, path, parameters and body, response status counts, elapsed times, row counts, errors and before/after PostgreSQL failure counters. Response bodies and authentication headers are not saved

The first request is reported separately from repeated requests. This is not a cold-cache claim: previous cases, fixture creation and previous runs can warm database buffers or application caches. Restart the disposable proxy between independent runs when measuring process-cache recovery

Repeated p50 is reported for successful samples, p95 only with at least 20 and p99 only with at least 100. Failures never become fast successful samples. The current assertions cover HTTP success, response shape, nonempty fixture results, cursor correctness, key recovery, timeout and zero failed PostgreSQL operations. There are no CPU, Redis-call or PostgreSQL-call-count limits and no latency ceiling beyond the request timeout

Validate a small restore first. Small-database numbers validate the runner only, not full-size performance. Run the same cases against the completed full-size snapshot before publishing large-database results
