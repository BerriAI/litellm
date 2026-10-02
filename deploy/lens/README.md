# Lens worker

Lens reviews recorded activity and saves evidence-linked findings in the LiteLLM dashboard under Observability, Lens (`/ui/lens/`)

## Start a worker

Upgrade your existing LiteLLM proxy to a release that includes Lens with PostgreSQL, agent tracing (`general_settings.tracing: {store: clickhouse}`), and ClickHouse configured through `CLICKHOUSE_URL` and a separate SELECT-only `CLICKHOUSE_READER_URL`. Enable the ClickHouse callback and request/response logging to analyze LLM requests. Lens can only inspect content you actually retain

In Lens, click **Set up analysis**, choose an existing virtual key or **Create worker key**, then **Generate setup command**. The LiteLLM address is filled in for you; change it only if the server running Docker needs a different network address. Copy the command and run it on your server. The dialog changes to **Analyzer connected** when the container checks in

The command already contains the compatible worker image and one worker token. The selected virtual key stays on the proxy; its secret is never sent to the worker. No source checkout, environment file, or second LiteLLM deployment is needed. Keep the command private because it includes the token. The LiteLLM release provides the dashboard and APIs; the container only runs background analysis

The dashboard and Compose file pin a verified worker image by digest. The image uses Linux amd64, and the generated command selects that platform. Worker image releases are independent of proxy releases: update the pinned image when changing their API contract. CI also publishes immutable commit tags for reproducible builds

For deployments managed with Compose, download `compose.yaml` and provide `LITELLM_URL` and `LENS_WORKER_TOKEN` in an environment file. Its default image is already selected:

```bash
docker compose --env-file /path/to/lens.env -f compose.yaml up -d
```

Developers can build locally with `LENS_WORKER_IMAGE=litellm-lens-worker:local docker compose -f deploy/lens/compose.yaml -f deploy/lens/compose.build.yaml up -d --build`

The generated command gives the worker 1 GiB of temporary memory-backed storage, shared across parallel reviews. Change `size=1g` in the Docker command or set `LENS_WORKER_TMP_SIZE` with Compose to fit your server and workload. A storage failure marks the scan as failed, cleans up temporary traces, and leaves the worker available for other scans; it does not silently truncate the review. Existing workers must be recreated with the new image and mount options

The worker needs outbound HTTPS access to LiteLLM. It needs no inbound ports, provider keys, direct database access, or GPU. The proxy calls your selected model through its normal virtual-key authorization and inference pipeline; trace content reaches that model provider. Use a model with JSON output support and known token prices. One worker handles one scan at a time and can serve multiple lenses. For more throughput, start another worker with a separate credential

If your deployment restricts `allowed_ips`, allow the worker's address. For workers behind a reverse proxy with `use_x_forwarded_for: true`, also configure `mcp_trusted_proxy_ranges` with that proxy's CIDRs and, when needed, `mcp_xff_num_trusted_hops`. Lens reuses these existing trusted-proxy settings. Forwarded addresses without an established trust boundary are rejected by the allowlist; accepting them would let a worker impersonate an allowed address

V1 setup, manual runs, feedback, and worker credentials are restricted to proxy administrators. Proxy-admin viewers can inspect results. Regular user and team keys cannot access the Lens API. Worker credentials can serve the administrator’s lenses. Revoke it in the connection dialog when retiring a worker. Redeploy the worker alongside proxy upgrades so their API versions match

## Configure a lens

Choose agent runs, individual LLM requests, or both. The matching-activity preview updates as you choose an application (the recorded OpenTelemetry service.name) or, for request activity, a LiteLLM model group and add metadata conditions. It shows run names, timestamps, and trace IDs; open a run to inspect its original steps before starting analysis. Suggestions come from up to 100 recent executions and may not include every recorded attribute. You can enter other exact keys and values. Leave service and filters blank for all activity your account can access. Filters are exact key/value matches, combined with AND. Trace filters match span or resource attributes on the same span. Request filters match logged metadata, including caller metadata stored under `requester_metadata`; `tag=value` matches request tags. `swarm=research` works only if your instrumentation records that attribute

Describe how the agent should behave and optionally add specific checks. Select the lookback window, team and metadata, then choose the percentage to review and an optional maximum. **100% with no maximum selects every matching run**. The preview pages through all matching activity and lets you select particular runs. Percentage sampling uses a stable hash order, rounds up, and applies the optional maximum after the percentage

Choose your analysis model, parallelism and monthly budget. Parallelism controls simultaneous model calls, not the number of runs selected. New lenses run once by default. Turn on monitoring to repeat the same setup at a custom interval. **Run now** uses the same saved settings immediately, including the same lookback window and sampling. Every scan recalculates the window, so overlapping windows can review the same activity again. Duplicate a lens when you want a separate investigation without changing an existing monitor

Pausing stops future scheduled scans; cancel the active scan separately if needed. The worker polls every 10 seconds; creating a lens or clicking Run now queues a scan, and due schedules are queued when the worker polls. Scans for the same lens never overlap, and its next interval starts after completion. Closing the browser does not stop the worker. Configuration edits apply to the next scan. A running scan retains its settings and selected execution IDs across retries

## Read the results

Needs attention shows issues, highest priority first. Patterns contains useful trends and successful behavior that may not need a fix. Each finding starts with a short explanation and a next step when useful. Expand the limitations for uncertainty and counterexamples. Evidence is grouped by run and collapsed until you need it; each quote opens the original step

Use the batch selector or Scans tab to reopen previous results. Each batch keeps its own findings, settings, selected runs, coverage and cost. Older batches created before snapshot support remain available through accumulated findings. The Runs tab lists the selected batch's sample and can filter per-run observations, including runs without an observed issue and runs with insufficient evidence. These observations precede the final evidence investigation. Linked-run counts on findings include cited counterexamples, so they are not failure counts

Choose **This is expected** and explain why to teach later scans about acceptable behavior. Feedback is kept with the lens and included in subsequent reviews. It does not alter historical evidence or exempt different problems

## What a scan does

The proxy selects executions received or updated within the configured lookback window, with a two-minute settling period. Older rows without receipt timestamps use execution end time. Overlapping scans do not increment a finding's occurrence count for the same execution ID

A trace is spans sharing a trace ID within one team, not an automatically reconstructed conversation session. Requests are individual LLM calls. When both sources are enabled, requests correlated to a recorded span by response ID are excluded to reduce double counting

The worker reviews the selected executions in parallel. It pages through their recorded spans and gives the first reviewer a catalog, task and outcome excerpts. The reviewer can read more original content to resolve uncertainties. Large catalogs and groups of observations are processed in bounded context windows, with every page available. Grouping retains supporting run IDs in code, so a pattern occurring thousands of times does not require a model to repeat thousands of IDs. Candidate investigators can page through supporting observations, other runs and original evidence

There is no fixed total run, span, candidate or investigation-turn cutoff. Repeated or empty evidence requests stop a stalled investigation. Context windows, the configured budget, available model capacity and recorded evidence still bound practical work. The dashboard reports completed work and gaps. The investigator has no shell, browsing, code-editing or production-action tools

Each model response must match a bounded JSON schema. A malformed response gets one repair attempt through the same budget controls; repeated invalid output fails the scan. Both the worker and proxy validate quoted evidence. Findings retain exact quotes and open the source trace or request. Resolve a finding after a fix, or dismiss it with a reason. A resolved finding reopens when new execution IDs support the same pattern; dismissed findings remain dismissed

Coverage distinguishes eligible, sampled, reviewed, partial, and unassessable executions. Findings describe observations in the sample, not population-wide success rates or proven causes. A root span does not prove that a trace contains every expected span. Long, missing, redacted, or expired content limits the conclusions

## Operations and limits

PostgreSQL stores configurations, findings and all scan history, returned in pages of 50 jobs. Workers claim jobs with optimistic concurrency and a five-minute lease, renewed every 30 seconds. A disconnected job can be reclaimed up to three times. Cancellation stops subsequent work; a model call already in flight may finish and incur cost

Before every model call, Lens reserves a conservative amount against the monthly lens budget. Successful calls reconcile to reported cost where pricing is available. Interrupted calls retain their reservation because the provider may have charged. A scan stops when the next reservation would exceed the limit, so it can stop with some budget remaining. Both the Lens budget and the selected virtual key’s budgets, model permissions, and rate limits apply. Analysis spend appears under that key in Virtual Keys and normal request logs, with Lens, scan, and worker IDs in request metadata. Analysis prompts and responses are redacted from spend logs; source traces and findings remain available through the administrator-only Lens API. Existing workers need a billing key assigned in **Set up analysis** before they can resume

V1 requires ClickHouse for both sources. It does not reconstruct sessions from unrelated trace IDs, guarantee exhaustive reviews, cache all per-execution observations across scans, or automatically fix agent code. Trace contents can change as late spans arrive, even though a job's selected IDs are fixed. Findings should be reviewed by a person before acting on them


## API access

The UI and API use the same scan lifecycle. Authenticate with a proxy administrator credential for writes, or a proxy-admin viewer credential for reads. Worker credentials are only for worker operations

```bash
curl "$LITELLM_URL/lens" -H "Authorization: Bearer $LITELLM_API_KEY" \
  -H 'Content-Type: application/json' -d '{
    "name": "Research quality", "model": "your-model-alias",
    "context": "Answer the requested question using cited, retrieved evidence.",
    "source": "traces", "lookback_hours": 24,
    "sample_percent": 100, "sample_size": null, "concurrency": 8,
    "enabled": true, "interval_minutes": 1440, "monthly_budget": 50
  }'

curl "$LITELLM_URL/lens/$LENS_ID/runs" -X POST \
  -H "Authorization: Bearer $LITELLM_API_KEY" -H 'Content-Type: application/json' -d '{}'

curl "$LITELLM_URL/lens/$LENS_ID/runs?offset=0" -H "Authorization: Bearer $LITELLM_API_KEY"
curl "$LITELLM_URL/lens/$LENS_ID/runs/$BATCH_ID" -H "Authorization: Bearer $LITELLM_API_KEY"
```

Creation queues the first batch. Posting to `/lens/{id}/runs` queues another, or returns the existing active batch. The run response contains its ID under `jobs[0].id`. Poll the batch URL for status, findings and assessments. List responses omit large result payloads; request a batch to retrieve them. Supply an optional complete `settings` object on the runs POST for a one-off override; the saved lens stays unchanged. Selection accepts `team_id`, exact `filters`, and opaque `execution_ids` returned by `/lens/preview/sample`. Preview accepts `offset` and `as_of` to keep the time window fixed while paging. Feedback uses `PATCH /lens/{id}/findings/{finding_id}` with `status` and `reason`

## Quality evaluation

Run the checked-in cases against a configured real model. Expected labels are used only for scoring, never passed to the model. Dev and held-out cases include missing outcomes, failed tools, recovery, handoffs, unsupported claims, repeated work, long evidence and prompt injection. The background option adds clean arithmetic traces to test rare-issue discovery at scale; those repeated synthetic cases do not establish accuracy on every production workload

```bash
python -m tests.proxy_behavior.lens.evaluate --api-base "$LITELLM_URL" \
  --model your-model-alias --split all --background 1000 --concurrency 16 \
  --output /tmp/lens-quality.json
```

Set `LITELLM_API_KEY` privately. This makes paid model calls. Inspect missed and unexpected per-run labels, final findings and coverage; do not equate a passing dataset with guaranteed detection on arbitrary traces

The worker uses temporary disk space for trace content while reviewing it, and removes those files after each review. The Docker command supplies a writable temporary mount while keeping the application filesystem read-only

To check that accepted behavior stays accepted without hiding new problems, run the evaluator with `--dataset tests/proxy_behavior/lens/feedback_cases.json`. Reports include elapsed time, model call count, reported cost when the proxy provides it, missed checks, unexpected checks, and inconclusive candidates

## Upgrading from the original Lens API

The Lens API now uses `/lens` instead of `/engine`, list responses use `lenses`, and worker claims use `lens_id`. Upgrade the proxy and recreate every worker with the image shown by the upgraded dashboard before starting new scans. Update API clients to the new paths and response fields. Old worker images cannot poll the renamed API

Stop workers and let active scans finish before upgrading. Deploy proxy instances together: older proxies cannot use the renamed database tables. The schema migration renames the three Lens tables and the run-history identifier column in place, preserving saved investigations, findings, history, worker credentials, and billing assignments. Existing migration files retain their original names and checksums

Upgrades using `--use_prisma_db_push` stop before schema changes if any legacy Lens table exists, preventing Prisma from dropping saved data. Apply `litellm-proxy-extras/litellm_proxy_extras/migrations/20261001100000_rename_lens/migration.sql` to the configured database schema before retrying. Deployments already using migration history can instead start without `--use_prisma_db_push` to apply the shipped migration normally. Fresh databases and databases already using the renamed tables can continue using database push
