# Lens worker

Lens reviews recorded activity and saves evidence-linked findings in the LiteLLM dashboard under Observability, Lens (`/ui/lens/`)

## Install the release stack

Each stable, RC, and dev release containing Lens publishes the worker at the same version on GHCR and Docker Hub. Use the [LiteLLM releases page](https://github.com/BerriAI/litellm/releases) to select a version that includes the coordinated worker release

For a new local installation, install Docker with Compose, download the two release files, and create a private environment file. Replace `X.Y.Z` with the release version, without `v` (RCs use `X.Y.Z-rc.N`)

```bash
mkdir litellm-lens
cd litellm-lens
LENS_RELEASE=X.Y.Z
curl -fSLo compose.yaml "https://raw.githubusercontent.com/BerriAI/litellm/v${LENS_RELEASE}/deploy/lens/stack.yaml"
curl -fSLo config.yaml "https://raw.githubusercontent.com/BerriAI/litellm/v${LENS_RELEASE}/deploy/lens/config.yaml"
umask 077
printf 'LITELLM_VERSION=%s\nLITELLM_MASTER_KEY=sk-%s\nLITELLM_SALT_KEY=sk-%s\n' \
  "$LENS_RELEASE" "$(openssl rand -hex 32)" "$(openssl rand -hex 32)" > .env
printf 'POSTGRES_PASSWORD=%s\nCLICKHOUSE_PASSWORD=%s\n' \
  "$(openssl rand -hex 32)" "$(openssl rand -hex 32)" >> .env
docker compose up -d
```

Open `http://localhost:4000/ui/`, log in as `admin` with `LITELLM_MASTER_KEY` from `.env`, and add a model in the dashboard. In Lens, select **Connect worker**, choose that model and a monthly budget, then **Get install command**. Expand **Using Docker Compose or Helm?**, copy the worker token, and add `LENS_WORKER_TOKEN=<token>` to `.env`

```bash
docker compose --profile lens up -d
```

The stack starts LiteLLM, PostgreSQL, ClickHouse, and the worker from published images. The dashboard shows **Worker connected**. The worker has a limited token, no database credentials, and no provider keys. The stack exposes only the dashboard on localhost; use your normal ingress and managed databases for a public production deployment

Keep `.env` private and preserve its salt key. Keep both named database volumes. To upgrade, wait for active investigations to finish, stop the worker, change only `LITELLM_VERSION`, then pull and recreate the stack:

```bash
docker compose --profile lens stop lens-worker
# Update LITELLM_VERSION in .env to the new release
docker compose --profile lens pull
docker compose --profile lens up -d
```

This preserves your investigations, findings, model credentials, and worker token. Never use `down -v` during an upgrade. If moving from an existing installation, keep its databases and add the standalone worker instead of creating an empty replacement stack

## Helm

The componentized `helm/litellm` chart includes an optional Lens worker. Configure PostgreSQL and ClickHouse as usual, install the chart, then obtain a limited worker token from Lens setup. Store it in a Kubernetes Secret and enable the worker in your values:

```yaml
lensWorker:
  enabled: true
  tokenSecret:
    name: litellm-lens-worker
    key: token
```

Published release charts pin the worker's approved image digest. Source charts without a digest default to the chart's application version. The chart connects the worker to the backend service. Keep these values and the Secret when upgrading the chart so the gateway and worker upgrade together. `lensWorker.replicaCount` controls simultaneous investigations. To use a private registry or external proxy, set `lensWorker.image.repository`, `lensWorker.image.digest` (or `tag` for a source build), and `lensWorker.url`. A digest takes precedence over the tag. The dashboard uses the chart's worker image for standalone install commands too

## Standalone worker

Upgrade your existing LiteLLM proxy to a release that includes Lens with PostgreSQL and agent tracing. Configure one ClickHouse URL for trace writes, bounded reads, and Lens queries:

```yaml
general_settings:
  tracing:
    store:
      type: clickhouse
      url: os.environ/CLICKHOUSE_URL
      retention_days: 14
```

The URL, database, and retention settings can also come from `CLICKHOUSE_URL`, `CLICKHOUSE_DATABASE`, and `AGENT_TRACING_RETENTION_DAYS` when omitted from YAML. A YAML value wins when both are set. The database defaults to `litellm`. `retention_days` defaults to 14 and applies to both traces and spend logs

Retention changes require a proxy restart. ClickHouse removes expired rows during background merges, not immediately at startup. Enable request/response logging to analyze LLM requests. Lens can only inspect content you actually retain

In **Lens > Investigations**, click **Connect worker**, choose an analysis model and monthly limit, then **Get install command**. Use **Advanced options** to select an existing virtual key or change the proxy URL if the server running Docker needs a different network address. Copy the command and run it on your server. The dashboard shows **Worker connected** when the container checks in

The command already contains the compatible worker image and one worker token. The selected virtual key stays on the proxy; its secret is never sent to the worker. No source checkout, environment file, or second LiteLLM deployment is needed. Keep the command private because it includes the token. The LiteLLM release provides the dashboard and APIs; the container only runs background analysis

The dashboard selects the worker image matching the running gateway release. Release images support Linux amd64 and arm64. CI also publishes `:sha-<commit>` development images; use those only with a gateway built from the same commit and release tag

After upgrading the gateway, update the worker image and redeploy it while keeping its proxy URL and token. Existing containers do not update automatically. If an investigation reports a worker compatibility error, update the image before retrying

For deployments managed with Compose, download `compose.yaml` and provide `LITELLM_URL`, `LENS_WORKER_TOKEN`, and `LITELLM_VERSION` (without `v`) in a private environment file. To use another registry, set `LENS_WORKER_IMAGE` to the compatible image instead of setting a version:

```bash
docker compose --env-file /path/to/lens.env -f compose.yaml up -d
```

To work on Lens itself, `make lens-dev` runs the proxy, a worker from source and the hot-reload dashboard together; set `LENS_DEV_PROXY_PORT` / `LENS_DEV_UI_PORT` to move them off 4000/3000. For a local container build, set `LENS_WORKER_IMAGE=litellm-lens-worker:local` and `LITELLM_RELEASE_TAG` to the gateway's release tag, then use `docker compose -f deploy/lens/compose.yaml -f deploy/lens/compose.build.yaml up -d --build`

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

## Local development

`make lens-dev ARGS=--seed` starts the full dev stack. The live dashboard is at `http://localhost:3000/ui/lens/`, with login at `http://localhost:3000/ui/login/`. Next.js forwards API requests to the proxy on port 4000, so login and navigation stay in the live UI and edits hot-reload

The default is Next.js dev with no production build (`LENS_DEV_BUILD_UI=0`). Set `LENS_DEV_BUILD_UI=1` when you also want a fresh static dashboard at `http://localhost:4000/ui/`. Build output goes to `.lens-dev/logs/ui-build.log`; a failed build stops startup. Both modes keep the live dashboard on port 3000. Startup checks the live login route before seeding and fails with the UI log path if Next.js exits. `LENS_DEV_STARTUP_TIMEOUT_SECONDS` controls startup readiness retries (default 300; `LENS_DEV_READINESS_REQUEST_TIMEOUT_SECONDS` caps each HTTP probe, default 5)

For local fixture data, run `make lens-dev ARGS=--seed`. Use `make lens-dev ARGS="--seed large"` for 2,000 fixture copies, over one million spans and linked request logs. To seed a running stack without restarting it, use `make lens-dev ARGS="--seed-only --seed large --copies 100"`. The default profile replays one copy of every checked-in capture through authenticated `/v1/traces`, including failures, retries, streaming and multiple agent frameworks. Large seeds use the same parser and compressed ClickHouse writer in batches of four copies, and write matching request logs to PostgreSQL. The first and last batches verify linked spend totals through the proxy

Seeds append fresh IDs on every invocation and spread copies over recent timestamps. Restarts without `SEED` do not add data. Lens excludes activity received in the last two minutes, so wait two minutes after seeding before checking investigation previews. `LENS_DEV_SEED_COPIES` overrides total copies, and `LENS_DEV_SEED_BATCH_COPIES` overrides copies per bulk insert (default 4, about 2,000 spans). Start with four or fewer on a constrained machine. Larger batches still respect the existing ClickHouse insert size limit; each capture is decoded separately within the OTLP safety budget. Large seeds test data volume and pagination, rather than concurrent ingestion throughput or review accuracy. They can use substantial disk space; adjust `--copies` for your machine. Seeding expects the generated local tracing configuration. The old `run_tracing_proxy_local.sh --seed` command forwards to Lens dev, using its ports and saved master key

Local ingestion limits are explicit and configurable. Set OTLP and ClickHouse variables before starting the proxy and seeder so both processes use the same settings. Invalid, zero and negative values fail instead of silently falling back. Changing these limits does not require rebuilding Rust

| Environment variable | Default | Controls |
| --- | --- | --- |
| `LENS_DEV_SEED_COPIES` | 1 default, 2000 large | Total fixture copies |
| `LENS_DEV_SEED_BATCH_COPIES` | 4 | Copies per bulk insert |
| `LENS_DEV_SEED_TIMEOUT_SECONDS` | 120 | Seeder HTTP timeout |
| `OTLP_MAX_BODY_BYTES` | 16777216 | HTTP body and decompressed payload bytes |
| `OTLP_MAX_CONCURRENT_INGESTS` | 2 | Concurrent proxy ingestion requests |
| `OTLP_MAX_ATTRIBUTE_VALUE_BYTES` | 65536 | Stored attribute/content bytes |
| `OTLP_MAX_DECODE_DEPTH` | 32 | Nested decode depth |
| `OTLP_MAX_DECODE_NODES` | 65536 | JSON values or protobuf fields per export |
| `OTLP_MAX_SPANS` | 4096 | Spans per export |
| `OTLP_MAX_ATTRIBUTES` | 256 | Attributes per resource, scope, span, event or link |
| `OTLP_MAX_EVENTS` | 256 | Events per span |
| `OTLP_MAX_LINKS` | 256 | Links per span |
| `OTLP_MAX_DECODED_SPAN_BYTES` | 16777216 | Decoded span allocation budget |
| `CLICKHOUSE_TRACE_MAX_INSERT_BYTES` | 67108864 | Encoded trace or spend insert bytes |
| `CLICKHOUSE_INSERT_TIMEOUT_SECONDS` | 30 | ClickHouse insert HTTP timeout |

The wire parsers also enforce their library recursion limits (128 levels for JSON, 100 for protobuf). Raising the configured depth does not remove those parser limits. Bulk seeding parses each capture separately, keeping the per-export limits distinct from the bulk insert limit. Use smaller batches if an insert exceeds its byte budget. For example, `LENS_DEV_SEED_COPIES=100 LENS_DEV_SEED_BATCH_COPIES=2 make lens-dev ARGS="--seed large"`

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


## Release compatibility

Released gateway and worker images carry `LITELLM_RELEASE_TAG`. A worker announces its release and protocol before claiming an investigation. A mismatch returns HTTP 409 with the required image, leaving queued investigations untouched. During a rolling upgrade, workers wait for a gateway from their release

The dashboard reads its image from the running gateway. `LENS_WORKER_IMAGE` overrides the registry/image for private deployments. Worker-only Compose accepts `LITELLM_VERSION` (without `v`) or an explicit `LENS_WORKER_IMAGE`. Release workers are available as `ghcr.io/berriai/litellm-lens-worker:vX.Y.Z` and `docker.io/litellm/litellm-lens-worker:vX.Y.Z`, including matching RC/dev suffixes, on amd64 and arm64

For source development, use `make lens-dev`, which gives the proxy and source worker the same commit identity. For custom containers, build both from the same checkout with `--build-arg LITELLM_RELEASE_TAG=sha-$(git rev-parse HEAD)` and set the proxy's `LENS_WORKER_IMAGE` to the worker image you built. An unlabelled custom build refuses worker setup and claims instead of guessing from the Python package version. Normal package-index installations use their installed release version

The hourly development pipeline pins all component images to the same selected commit and publishes its chart only after every build and worker smoke test succeeds. The public commit-tagged worker workflow publishes to `ghcr.io/berriai/litellm-lens-worker-dev` on Lens-related changes, so an arbitrary `main` commit may require building your own pair; do not substitute the newest available worker


## Worker dependencies

The worker uses the same digest-pinned Wolfi base and Python version as the component images. Python dependencies and their hashes are locked in `deploy/lens/requirements.lock`. To update them, edit `deploy/lens/requirements.in`, then run `uv pip compile --universal --python-version 3.13 --generate-hashes --no-emit-index-url deploy/lens/requirements.in -o deploy/lens/requirements.lock`. The image installs only the locked wheels with hash verification. CI builds and scans both native architectures
