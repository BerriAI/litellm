# Admin UI journey latency on the 1TB SpendLogs restore

Run `test_ui_journeys.py` against a dedicated proxy, its Admin UI and a disposable database restored from the metadata-only performance seed. A headless Chromium logs into the dashboard and performs the journeys below, timing each one from the user's click or navigation until the data is on screen. No LLM is called. The test does not look at which routes, SQL or tables served the page, so it stays valid whether the backend gets faster queries or new aggregate tables

## Journeys

| name | what the user does | timer stops when |
| --- | --- | --- |
| `logs_open_24h` | opens Logs (default Last 24 Hours) | request rows are rendered and the refresh control is idle |
| `logs_last_7_days` | switches the time range to Last 7 Days | same |
| `logs_next_page` | clicks the next page control | same |
| `logs_filter_key_hash` | filters by the seeded key hash and applies | same |
| `logs_filter_key_alias` | picks the first key alias in the filter drawer and applies | same |
| `logs_filter_team` | picks the first team in the filter drawer and applies | same |
| `logs_filter_model` | picks the first model in the filter drawer and applies | same |
| `usage_open` | opens Usage | summary tiles show dollar values with no skeletons left |
| `usage_last_30_days` | picks Last 30 days in the Usage date picker | same |
| `usage_team_view` | switches the Usage view to Team Usage | same |
| `usage_customer_view` | switches the Usage view to Customer Usage | same |
| `usage_model_activity_30_days` | on the Model Activity tab, picks Last 30 days | same |
| `cache_open` | opens Caching | the Cache Hits tile is visible and the Cache Hits vs API Requests chart has bars |
| `models_open` | opens Models + Endpoints | model rows are rendered and the loading state is gone |
| `keys_open` | opens Virtual Keys | key rows are rendered and the loading state is gone |

The mix follows what customers report as slow: the Logs page on `LiteLLM_SpendLogs`, the Usage page on the `LiteLLM_Daily*Spend` rollups (30 day ranges, Customer and Model views), and the Models and Keys pages.

## Seeded rows

The snapshot carries the SpendLogs rows and their Daily* rollups. `setup/ui_query_fixture.py` checks that `LiteLLM_DailyUserSpend` and `LiteLLM_DailyTeamSpend` have rows in the week before the anchor, then fills the tables the snapshot does not have so the Keys, Models and filter journeys see realistic counts: 100 teams (`perf-team-N`), 2,000 users (`perf-user-N`), 5,000 keys whose token hashes and `perf-key-N` aliases match the seeded log rows, and 200 DB models pointed at the perf mock provider. Every insert is `ON CONFLICT DO NOTHING`, so reruns on the same restore are no-ops. Nothing here runs outside the disposable restore

Each journey is measured once cold, then `E2E_UI_JOURNEY_REPEATS` more times (default 9) from the same starting state, inside one logged-in tab whose clock is pinned to the seed anchor so relative windows land on seeded rows. The browser timezone is UTC to match the seed

## Safety

The seeded source is never the target. `E2E_UI_PERF_DISPOSABLE=1`, `E2E_RESET_SPEND_LOGS=0` and a hostname matching `db-perf-perf-<id>.litellm.svc` are all required, and in-flight SQL is cancelled only on that disposable restore between journeys (see `setup/ui_query_isolation.py`)

## Running

The componentized perf stack serves the dashboard behind the same router as the proxy, so `LITELLM_PROXY_URL` covers `/ui`. Set `E2E_UI_BASE_URL` only if the UI lives elsewhere. The admin login uses `E2E_UI_USERNAME` (default `admin`) and `E2E_UI_PASSWORD` (default the master key)

```bash
export E2E_PERF_TESTS=1 E2E_UI_PERF_DISPOSABLE=1 E2E_RESET_SPEND_LOGS=0
export DATABASE_URL=postgresql://...@db-perf-perf-<id>.litellm.svc:5432/litellm
export E2E_DATABASE_METRICS_URLS=http://<backend-pod>:4000/metrics
export E2E_ARTIFACT_DIR=/tmp/e2e-artifacts
uv run pytest tests/e2e/perf/test_ui_journeys.py --reruns 0 --junitxml="$E2E_ARTIFACT_DIR/junit.xml" -q
```

`E2E_UI_JOURNEY_TIMEOUT_SECONDS` (default 120) bounds one journey. A journey that never settles is recorded as a `timeout` sample and a journey whose page rendered empty or threw in the browser as a `ui` sample. The test fails when any sample failed, or when the p50 or p90 across all samples is over the journey's `Ceiling` in `UI_JOURNEY_LIMITS` (`setup/ui_journeys.py`, 10s/20s for Logs journeys, 20s/40s for Usage and Caching, 5s/10s for the Models and Keys tables, generous until the first 1TB run sets real numbers). The artifacts are written before the assertion so a failed run still ships its samples

## Artifacts

Each JUnit testcase `test_ui_journey_latency[<name>]` carries `perf_ui_journeys`, `perf_ui_failures`, `perf_ui_seconds_p50`, `perf_ui_seconds_p90`, `perf_ui_seconds_max`, `perf_ui_first_seconds`, `perf_ui_repeated_seconds_p50`, `perf_ui_repeated_seconds_max` and `perf_ui_database_failures`. `ui-journey-<name>.json` holds every sample with its on-screen row count and `ui-isolation-<name>.jsonl` the SQL isolation evidence

The Buildkite `ui_queries` lane restores the pinned snapshot, waits for migrations and indexes, runs these journeys with no retries and uploads the JUnit and JSON artifacts. Validate selectors on a small restore first, then take numbers from the completed 1TB snapshot

To measure a released version on the same snapshot, create the litellm-perf build with `perf_gateway_version=<tag>` (for example `v1.106.1`) next to `litellm_sha`. The gateway, backend and the restore's schema upgrade then come from `ghcr.io/berriai/litellm-{gateway,backend,migrations}:<tag>`, where the migrations image runs `prisma migrate deploy` and builds the request-log indexes the way a customer upgrade does, while the runner and these journeys come from `litellm_sha`. The fixture skips its upgrade-worker wait when `_LiteLLMPerfUpgrade` is absent. The snapshot's schema is v1.105.1 and the restore only migrates forward, so older tags cannot be measured

Each sample in `ui-journey-<name>.json` also records the `status` and `request` (method and path) of the data request the page issued, the first non-2xx one when there is one, so a failed sample names what the gateway answered. A failed sample with a 4xx or 5xx status is counted as `perf_ui_error_http_<status>` instead of `perf_ui_error_ui`. Each journey is also bounded by a wall clock of `E2E_UI_JOURNEY_TIMEOUT_SECONDS x (repeats + 2)`, so one hung browser call ends the journey with a `wall_clock` sample instead of eating the lane.
