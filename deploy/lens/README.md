# Lens worker

Lens reviews recorded activity and saves evidence-linked findings in the LiteLLM dashboard under Observability, Lens (`/ui/lens/`)

## Start a worker

Upgrade your existing LiteLLM proxy to a release that includes Lens with PostgreSQL, agent tracing (`general_settings.tracing: {store: clickhouse}`), and ClickHouse configured through `CLICKHOUSE_URL` and a separate SELECT-only `CLICKHOUSE_READER_URL`. Enable the ClickHouse callback and request/response logging to analyze LLM requests. Lens can only inspect content you actually retain

In Lens, click **Connect worker**, then **Generate setup command**. The LiteLLM address is filled in for you; change it only if the server running Docker needs a different network address. Copy the command and run it on your server. The dialog changes to **Worker connected** when the container checks in

The command already contains the compatible worker image and one worker token. No separate API key, source checkout, environment file, or second LiteLLM deployment is needed. Keep the command private because it includes the token. The LiteLLM release provides the dashboard and APIs; the container only runs background analysis

The dashboard and Compose file pin a verified worker image by digest. The image uses Linux amd64, and the generated command selects that platform. Worker image releases are independent of proxy releases: update the pinned image when changing their API contract. CI also publishes immutable commit tags for reproducible builds

For deployments managed with Compose, download `compose.yaml` and provide `LITELLM_URL` and `LENS_WORKER_TOKEN` in an environment file. Its default image is already selected:

```bash
docker compose --env-file /path/to/lens.env -f compose.yaml up -d
```

Developers can build locally with `LENS_WORKER_IMAGE=litellm-lens-worker:local docker compose -f deploy/lens/compose.yaml -f deploy/lens/compose.build.yaml up -d --build`

The worker needs outbound HTTPS access to LiteLLM. It needs no inbound ports, provider keys, direct database access, or GPU. The proxy calls your selected model through its configured router; trace content reaches that model provider. Use a model with JSON output support and known token prices. One worker handles one scan at a time and can serve multiple lenses. For more throughput, start another worker with a separate credential

V1 setup, manual runs, feedback, and worker credentials are restricted to proxy administrators. Admin viewers can inspect results. Worker credentials can serve the administrator’s lenses. Revoke it in the connection dialog when retiring a worker. Redeploy the worker alongside proxy upgrades so their API versions match

## Configure a lens

Choose agent runs, individual LLM requests, or both. The matching-activity preview updates as you choose an application (the recorded OpenTelemetry service.name) or, for request activity, a LiteLLM model group and add metadata conditions. It shows run names, timestamps, and trace IDs; open a run to inspect its original steps before starting analysis. Suggestions come from up to 100 recent executions and may not include every recorded attribute. You can enter other exact keys and values. Leave service and filters blank for all activity your account can access. Filters are exact key/value matches, combined with AND. Trace filters match span or resource attributes on the same span. Request filters match logged metadata, including caller metadata stored under `requester_metadata`; `tag=value` matches request tags. `swarm=research` works only if your instrumentation records that attribute

Write a few questions, give context about a successful run, choose a model, and set the monthly limit and sample size. Choose an initial history window from 1 hour to 30 days, in hours or days. Creation queues the first scan over that window. New lenses run once by default; opt into background monitoring for a custom interval from 1 minute to 7 days, entered in minutes, hours, or days. **Analyze now** checks activity since the last successful scan; **Recheck the last 24 hours** revisits recent history. The runs API accepts `lookback_hours` from 1 to 720 for other historical windows

Pausing stops future scheduled scans; cancel the active scan separately if needed. The worker polls every 10 seconds; creating a lens or clicking Analyze now queues a scan, and due schedules are queued when the worker polls. Scans for the same lens never overlap, and its next interval starts after completion. Closing the browser does not stop the worker. Configuration edits apply to the next scan. A running scan retains its settings and selected execution IDs across retries

## Read the results

Needs attention shows issues, highest priority first. Patterns contains useful trends and successful behavior that may not need a fix. Each finding starts with a short explanation and a next step when useful. Expand the limitations for uncertainty and counterexamples. Evidence is grouped by run and collapsed until you need it; each quote opens the original step

The Runs tab lists the actual sample frozen for the latest scan. Linked-run counts on findings include cited counterexamples, so they are not failure counts. The Scans tab shows history and coverage. Existing findings retain their original wording; the shorter summaries apply to new analysis

## What a scan does

The proxy selects newly received or updated executions with a two-minute settling period and a five-minute overlap. Older rows without receipt timestamps use execution end time. Overlapping scans do not increment a finding's occurrence count for the same execution ID

A trace is spans sharing a trace ID within one team, not an automatically reconstructed conversation session. Requests are individual LLM calls. When both sources are enabled, requests correlated to a recorded span by response ID are excluded to reduce double counting

The worker screens a deterministic sample, at most the configured 1–500 executions. For each execution it reads up to 160 spans, with 8,000 characters per span section, and splits these into model calls. It consolidates observations across batches, then investigates at most 10 candidate patterns using up to five model turns each. The dashboard shows these three stages, completed work counts, and elapsed time; progress is based on the selected sample, not every eligible execution. The investigator can read more original content from the selected executions. It has no shell, browsing, code-editing, or production-action tools

Each model response must match a bounded JSON schema. A malformed response gets one repair attempt through the same budget controls; repeated invalid output fails the scan. Both the worker and proxy validate quoted evidence. Findings retain exact quotes and open the source trace or request. Resolve a finding after a fix, or dismiss it with a reason. A resolved finding reopens when new execution IDs support the same pattern; dismissed findings remain dismissed

Coverage distinguishes eligible, sampled, reviewed, partial, and unassessable executions. Findings describe observations in the sample, not population-wide success rates or proven causes. A root span does not prove that a trace contains every expected span. Long, missing, redacted, or expired content limits the conclusions

## Operations and limits

PostgreSQL stores configurations, findings and the latest 50 jobs. Workers claim jobs with optimistic concurrency and a five-minute lease, renewed every 30 seconds. A disconnected job can be reclaimed up to three times. Cancellation stops subsequent work; a model call already in flight may finish and incur cost

Before every model call, Lens reserves a conservative amount against the monthly lens budget. Successful calls reconcile to reported cost where pricing is available. Interrupted calls retain their reservation because the provider may have charged. A scan stops when the next reservation would exceed the limit, so it can stop with some budget remaining. Lens budgets are separate from virtual-key budgets; analysis calls use the proxy router directly

V1 requires ClickHouse for both sources. It does not reconstruct sessions from unrelated trace IDs, guarantee exhaustive reviews, cache all per-execution observations across scans, or automatically fix agent code. Trace contents can change as late spans arrive, even though a job's selected IDs are fixed. Findings should be reviewed by a person before acting on them
