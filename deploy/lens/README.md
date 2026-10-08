# Lens worker

Lens reviews recorded activity and saves evidence-linked findings in the LiteLLM dashboard under Observability, Lens (`/ui/lens/`)

## Install

Build LiteLLM and its worker from the same source commit with the same release identity. The worker runs separately and connects to your gateway using a limited worker token

### New local installation

Install Docker with Compose and Git. This builds LiteLLM and its worker from the same checkout and starts the existing local tracing stack:

```bash
git clone https://github.com/BerriAI/litellm.git
cd litellm
export LITELLM_RELEASE_TAG="sha-$(git rev-parse HEAD)"
export LENS_WORKER_IMAGE="litellm-lens-worker:${LITELLM_RELEASE_TAG}"
export OPENAI_API_KEY='sk-...'
docker build --build-arg LITELLM_RELEASE_TAG="$LITELLM_RELEASE_TAG" \
  -f deploy/lens/Dockerfile -t "$LENS_WORKER_IMAGE" .
docker compose -f docker/docker-compose.tracing.yml up -d --build
```

Open `http://localhost:4002/ui/` and sign in as `admin` with the key saved in `.lens-dev/master_key`. Go to **Lens > Investigations > Connect worker**, choose a model and monthly budget, then **Get install command**. Expand **Using Docker Compose or Helm?** and copy the worker token. In the same terminal, run:

```bash
export LITELLM_URL=http://litellm:4000
export LENS_WORKER_TOKEN='<paste-your-worker-token>'
docker compose -f docker/docker-compose.tracing.yml -f deploy/lens/compose.yaml up -d
```

The worker joins the gateway's Docker network, and the dashboard shows **Worker connected**. Save the token privately for restarts and upgrades

This stack is for local evaluation: it binds to localhost and uses development database credentials. For a hosted deployment, keep your normal database, keys, networking, and deployment process. Build both images from one source revision with the same `LITELLM_RELEASE_TAG`, publish the worker to your registry, and set `LENS_WORKER_IMAGE` on LiteLLM to that image

### Existing LiteLLM installation

Keep your deployment and PostgreSQL database. A working gateway/worker pair can stay as it is until you upgrade both. For a gateway built from source, use its exact commit and `LITELLM_RELEASE_TAG`; a release version or the latest commit on `main` is not a substitute for that source identity

The public development package is `ghcr.io/berriai/litellm-lens-worker-dev:sha-<full-commit>`. It publishes amd64 images on Lens-related changes, so an arbitrary source commit may have no image. Check the exact image exists before using it. If it is unavailable, your gateway uses a different release identity, or you need native arm64, build the worker from the gateway's checkout:

```bash
export LITELLM_RELEASE_TAG='<gateway-release-identity>'
export LENS_WORKER_IMAGE='<your-registry>/litellm-lens-worker:<your-image-tag>'
docker build --build-arg LITELLM_RELEASE_TAG="$LITELLM_RELEASE_TAG" \
  -f deploy/lens/Dockerfile -t "$LENS_WORKER_IMAGE" .
```

For a remote worker host, publish that image to a registry the host can pull from. Set the gateway's `LENS_WORKER_IMAGE` to the resulting image reference, restart the gateway using its normal deployment process, then copy its install command. Prefer the published image digest for hosted installations. Do not change the gateway's release identity just to accept another worker

For Kubernetes or Render, run the standalone worker using `LITELLM_URL` and `LENS_WORKER_TOKEN` from setup. Keep existing databases and secrets. The worker needs no inbound port.

## Helm

The componentized source chart at `helm/litellm` includes an optional Lens worker. Use the chart from the same checkout as your gateway and keep your component image overrides in your values. Configure PostgreSQL and ClickHouse as usual, install the chart, then obtain a limited worker token from Lens setup. Store it in a Kubernetes Secret and enable the worker in your values:

```yaml
lensWorker:
  enabled: true
  image:
    repository: <your-worker-image-repository>
    digest: sha256:<matching-worker-image-digest>
  tokenSecret:
    name: litellm-lens-worker
    key: token
```

Set the worker repository and digest explicitly to an image built from the gateway's source commit and release identity. The chart connects the worker to the backend service. Keep these values and the Secret when upgrading the chart and update the gateway and worker image overrides together. `lensWorker.replicaCount` controls simultaneous investigations. To use a private registry or external proxy, set `lensWorker.image.repository`, `lensWorker.image.digest` (or `tag` for a source build), and `lensWorker.url`. A digest takes precedence over the tag. The dashboard uses the chart's worker image for standalone install commands too

## Standalone worker

Start with a source deployment that includes Lens, PostgreSQL, and agent tracing, and prepare its matching worker as described above. Configure one ClickHouse URL for trace writes, bounded reads, and Lens queries:

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

The command already contains the compatible worker image and one worker token. The selected virtual key stays on the proxy; its secret is never sent to the worker. Once the matching image is available on the worker host, no second LiteLLM deployment is needed. Keep the command private because it includes the token. The LiteLLM release provides the dashboard and APIs; the container only runs background analysis

The dashboard uses the gateway's `LENS_WORKER_IMAGE` override when set. Public `:sha-<commit>` development images must match both the gateway commit and release identity. Build from source for the worker host's native architecture

After upgrading the gateway, update the worker image and redeploy it while keeping its proxy URL and token. Existing containers do not update automatically. If an investigation reports a worker compatibility error, update the image before retrying

For deployments managed with Compose, download `compose.yaml` and provide `LITELLM_URL`, `LENS_WORKER_TOKEN`, and an explicit `LENS_WORKER_IMAGE` in a private environment file:

```bash
docker compose --env-file /path/to/lens.env -f compose.yaml up -d
```

To work on Lens itself, `make lens-dev` runs the proxy, a worker from source and the hot-reload dashboard together; set `LENS_DEV_PROXY_PORT` / `LENS_DEV_UI_PORT` to move them off 4000/3000. For a local container build, set `LENS_WORKER_IMAGE=litellm-lens-worker:local` and `LITELLM_RELEASE_TAG` to the gateway's release tag, then use `docker compose -f deploy/lens/compose.yaml -f deploy/lens/compose.build.yaml up -d --build`

The generated command gives the worker 1 GiB of temporary memory-backed storage, shared across parallel reviews. Change `size=1g` in the Docker command or set `LENS_WORKER_TMP_SIZE` with Compose to fit your server and workload. Python reports storage failures to the reviewer and cleans up temporary files, so the reviewer can retry a smaller computation or report insufficient evidence. The worker remains available for other scans. Existing workers must be recreated with the new image and mount options

The worker needs outbound HTTPS access to LiteLLM. It needs no inbound ports, provider keys, direct database access, or GPU. The proxy calls your selected model through its normal virtual-key authorization and inference pipeline; trace content reaches that model provider. Use a model with JSON output support and known token prices. One worker handles up to three investigations concurrently and can serve multiple lenses. For more throughput, start another worker with a separate credential

If your deployment restricts `allowed_ips`, allow the worker's address. For workers behind a reverse proxy with `use_x_forwarded_for: true`, also configure `mcp_trusted_proxy_ranges` with that proxy's CIDRs and, when needed, `mcp_xff_num_trusted_hops`. Lens reuses these existing trusted-proxy settings. Forwarded addresses without an established trust boundary are rejected by the allowlist; accepting them would let a worker impersonate an allowed address

Setup, manual runs, feedback, and worker credentials are restricted to proxy administrators. Proxy-admin viewers can inspect results. Regular user and team keys cannot access the Lens API. Worker credentials can serve the administrator’s lenses. Revoke it in the connection dialog when retiring a worker. Redeploy the worker alongside proxy upgrades so their API versions match

## Configure a lens

Choose agent runs, individual LLM requests, or both. The matching-activity preview updates as you choose an application (the recorded OpenTelemetry service.name) or, for request activity, a LiteLLM model group and add metadata conditions. It shows run names, timestamps, and trace IDs; open a run to inspect its original steps before starting analysis. Suggestions come from up to 100 recent executions and may not include every recorded attribute. You can enter other exact keys and values. Leave service and filters blank for all activity your account can access. Filters are exact key/value matches, combined with AND. Trace filters match span or resource attributes on the same span. Request filters match logged metadata, including caller metadata stored under `requester_metadata`; `tag=value` matches request tags. `swarm=research` works only if your instrumentation records that attribute

Describe how the agent should behave and optionally add specific checks. Select the lookback window, team and metadata, then choose the percentage to review and an optional maximum. **100% with no maximum selects every matching run**. The preview pages through all matching activity and lets you select particular runs. Percentage sampling uses a stable hash order, rounds up, and applies the optional maximum after the percentage

Choose your analysis model, parallelism and monthly budget. Parallelism controls simultaneous model calls, not the number of runs selected. New lenses run once by default. Turn on monitoring to repeat the same setup at a custom interval. **Run now** uses the same saved settings immediately, including the same lookback window and sampling. Each scan recalculates the window and reuses completed reviews when the selected trace content, expected behavior, enabled checks and analysis model are unchanged. Budget, name and schedule edits preserve reuse. Duplicate a lens when you want a separate investigation without changing an existing monitor

Pausing stops future scheduled scans; cancel the active scan separately if needed. The worker polls every two seconds; creating a lens or clicking Run now queues a scan, and due schedules are queued when the worker polls. Scans for the same lens never overlap, and its next interval starts after completion. Closing the browser does not stop the worker. A running scan retains its analysis settings and selected execution IDs across retries. Budget edits apply to subsequent model calls, including those in an active scan

## Read the results

Needs attention shows issues, highest priority first. Patterns contains useful trends and successful behavior that may not need a fix. Each finding starts with a short explanation and a next step when useful. Expand the limitations for uncertainty and counterexamples. Evidence is grouped by run and collapsed until you need it; each quote opens the original step

Use the **Investigation run** selector or **History** to reopen previous results. Each run keeps its new or updated findings, settings, selected traces, coverage and cost. An unchanged rerun adds no findings; choose **All accumulated findings** to see saved findings across runs. The **Agent traces** tab lists the selected sample, including traces without an observed issue and traces with insufficient evidence. The tab is called **LLM requests** or **Traces and requests** for those activity types. Findings distinguish distinct affected traces from contributing investigation runs. Counterexamples remain visible as evidence without increasing the affected count. Merged finding links continue to resolve to the retained finding

Enter an explanation under **What should Lens remember?** and choose **This is expected** to dismiss expected behavior, **Mark resolved** after fixing an issue, or **Reopen** to reopen a resolved issue. These actions save the feedback together with the status; typing feedback alone neither saves it nor resolves the finding. Feedback informs later analysis and reconciliation without invalidating completed reviews. It stays with the finding when evidence recurs, does not alter historical evidence, and does not exempt different problems

## What a scan does

The proxy selects executions received or updated within the configured lookback window, with a two-minute settling period. Older rows without receipt timestamps use execution end time. Overlapping scans do not increment a finding's occurrence count for the same execution ID

A trace is spans sharing a trace ID within one team, not an automatically reconstructed conversation session. Requests are individual LLM calls. When both sources are enabled, requests correlated to a recorded span by response ID are excluded to reduce double counting

The worker reads complete selected trace content to compute fingerprints before making paid model calls. Completed review checkpoints are reused only within the same lens when both the complete content and investigation criteria match. Criteria are the expected behavior, enabled check IDs and instructions, and analysis model. Lens fingerprints these values rather than using the time of an unrelated settings edit. Scope and sampling changes preserve matching reviews for traces selected again; duplicating a lens starts an independent set of reviews. Initial reviews are confined to their assigned trace and retain observations, cited excerpts and metadata. Reviewers use catalog, read, search and optional Python tools; Python receives selected evidence as streamed input. Pending observation batches are grouped in parallel and investigated against the original evidence, then reconciled with saved findings. A recurring cause extends its existing finding, preserving feedback, earlier evidence and contributing run history

If every selected review is already incorporated into findings, the run completes with a reuse count, zero model calls and zero analysis cost, including when the monthly budget is exhausted. The run remains in history. A partially failed run preserves completed review checkpoints; pending grouping or investigation can still need paid model calls even when all selected trace reviews are reused. A trace without a complete matching checkpoint needs a review. Live progress distinguishes reviews eligible for reuse from reviews actually recorded; failed or cancelled runs report only the reuse they completed

There is no fixed total run, span, candidate or investigation-turn cutoff. Agents can replace their active conversation with working notes. If a request exceeds the configured model's context window, the worker compacts the conversation automatically and resumes with references to its archived tool history. Original evidence remains accessible through the gateway while it is available and retained. Tool results and working notes remain accessible during the investigation; character ranges make even a single oversized result readable in pieces. A review reports an error if the task or its replacement notes cannot fit. Context windows, the configured budget, worker resources and recorded evidence still bound practical work. The investigator has no browsing, code-editing or production-action tools

The live review drawer shows loading, trace review, parallel grouping, reconciliation and candidate investigation. It reports current model and tool operations, including context compaction, and retains tool-call counts on completed trace reviews. These counts describe attempted calls, not successful executions. This progress channel contains operation metadata, not Python code or tool output. Preliminary observations remain separate from final findings and their validated evidence

Each model response must match its JSON schema. A malformed response gets one repair attempt through the same budget controls. A session review that remains invalid or cannot fit marks that execution unassessable while other reviews continue. Broken evidence pagination or missing content pages return tool errors so the agent can inspect narrower spans or other evidence. Unreadable citations receive repair feedback. Verified excerpts remain available without fetching their source again. The affected source counts as partial, including failures discovered during later investigations, while the reviewer owns its assessment. Findings are published only after comparison with each other and saved findings finishes. If analysis stops before that comparison completes, completed trace reviews and their evidence remain saved for reuse, and the run retains its assessments and error. A later run can retry grouping and investigation. Runs with useful completed assessments or reconciled findings show partial results; total failures are marked failed. Transport errors, cancellation and budget exhaustion stop further analysis. Both the worker and proxy validate quoted evidence against original content. Per-run issue assessments follow supporting citations, including evidence found by another run's reviewer; counterexamples do not mark a run affected. Findings retain exact quotes and open the source trace or request. Resolve a finding after a fix, or dismiss it with a reason. A resolved finding reopens when new execution IDs support the same pattern; dismissed findings remain dismissed

Coverage distinguishes eligible, sampled, reviewed, partial, and unassessable executions. Findings describe observations in the sample, not population-wide success rates or proven causes. A root span does not prove that a trace contains every expected span. Long, missing, redacted, or expired content limits the conclusions

## Operations and limits

PostgreSQL stores configurations, findings and all scan history, returned in pages of 50 jobs. Workers claim jobs with optimistic concurrency and a five-minute lease, renewed every 30 seconds. A disconnected job can be reclaimed up to three times. Cancellation stops subsequent work; a model call already in flight may finish and incur cost

Lens checks which selected traces have reusable reviews before requesting model budget. Reuse needs no model call or reservation. New trace reviews and unfinished grouping or investigation can incur cost

Before each model call, Lens reserves a conservative allowance based on the input and permitted output. The summary separates settled monthly spend, unexpired reservations and available budget. With a $100 limit, $45 spent and $10 reserved, $45 is available for additional calls. Successful calls settle to recorded cost and release unused capacity. Failed or timed-out requests release their hold; abandoned holds expire after the proxy request timeout plus a grace period

Calls wait when concurrent reservations temporarily hold the remaining capacity. Waiting and model execution share the proxy's request timeout. If one request's allowance exceeds the unspent monthly budget, the error reports what the request needs and what remains. Reduce the deployment's output allowance or increase the limit. Paid analysis stops when the monthly limit is spent; completed reviews can still be reused. The monthly budget renews on the UTC calendar month

The assigned virtual key has independent budgets, model permissions and rate limits. Several investigations can share that key, so its limit can stop analysis even when one lens has budget left. Every worker needs a billing key assigned through worker setup or **Settings**. Analysis spend appears under that key in **Virtual Keys** and normal request logs, with Lens, run and worker IDs in request metadata. Analysis prompts and responses are redacted from spend logs; source traces and findings remain available through the administrator-only Lens API. Terminal budget, authentication or transport failures stop the scan after applicable retries and preserve completed checkpoints

Lens requires ClickHouse for both sources. It does not reconstruct sessions from unrelated trace IDs, guarantee exhaustive reviews, or automatically fix agent code. Trace contents can change as late spans arrive, even though a job's selected IDs are fixed. Changed content requires a matching review before reuse. Findings should be reviewed by a person before acting on them


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

For local fixture data, run `make lens-dev ARGS=--seed`. Use `make lens-dev ARGS="--seed large"` for 2,000 fixture copies spread over the last 24 hours, about 860,000 spans with linked request logs, plus three long sessions of roughly 1,150, 9,200 and 92,000 spans in a single trace for drawer paging and the oversized read path. Their trace IDs are printed at the end. To seed a running stack without restarting it, use `make lens-dev ARGS="--seed-only --seed large --copies 100"`. Every profile replays one copy of every checked-in capture through authenticated `/v1/traces`, including failures, retries, streaming and multiple agent frameworks, and verifies linked spend totals through the proxy. Large seeds then copy that first copy inside ClickHouse and PostgreSQL with `INSERT ... SELECT`, rewriting trace, span and call IDs so each copy keeps its own spend, and verify the last copy through the proxy

Seeds append fresh IDs on every invocation and spread copies over recent timestamps. Restarts without `SEED` do not add data. Lens excludes activity received in the last two minutes, so wait two minutes after seeding before checking investigation previews. `LENS_DEV_SEED_COPIES` overrides total copies. Large seeds test data volume and pagination, rather than concurrent ingestion throughput or review accuracy. They can use substantial disk space; adjust `--copies` for your machine. Seeding expects the generated local tracing configuration. The old `run_tracing_proxy_local.sh --seed` command forwards to Lens dev, using its ports and saved master key

Local ingestion limits are explicit and configurable. Set OTLP and ClickHouse variables before starting the proxy and seeder so both processes use the same settings. Invalid, zero and negative values fail instead of silently falling back. Changing these limits does not require rebuilding Rust

| Environment variable | Default | Controls |
| --- | --- | --- |
| `LENS_DEV_SEED_COPIES` | 1 default, 2000 large | Total fixture copies |
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

The wire parsers also enforce their library recursion limits (128 levels for JSON, 100 for protobuf). Raising the configured depth does not remove those parser limits.

## Quality evaluation

Run the checked-in cases against a configured real model. Expected labels are used only for scoring, never passed to the model. Dev and held-out cases include missing outcomes, failed tools, recovery, handoffs, unsupported claims, repeated work, long evidence and prompt injection. The background option adds clean arithmetic traces to test rare-issue discovery at scale; those repeated synthetic cases do not establish accuracy on every production workload

```bash
python -m tests.proxy_behavior.lens.evaluate --api-base "$LITELLM_URL" \
  --model your-model-alias --split all --background 1000 --concurrency 16 \
  --output /tmp/lens-quality.json
```

Set `LITELLM_API_KEY` privately. This makes paid model calls. Inspect missed and unexpected per-run labels, final findings and coverage; do not equate a passing dataset with guaranteed detection on arbitrary traces

The default workspace retrieves trace content on demand. Python calls have temporary scratch space that is removed after execution. The Docker command supplies a writable temporary mount while keeping the application filesystem read-only

To check that accepted behavior stays accepted without hiding new problems, run the evaluator with `--dataset tests/proxy_behavior/lens/feedback_cases.json`. Reports include elapsed time, model call count, reported cost when the proxy provides it, missed checks, unexpected checks, and inconclusive candidates

## Release compatibility

Gateway and worker builds carry the same `LITELLM_RELEASE_TAG`. A worker announces its release and protocol before claiming an investigation. A mismatch returns HTTP 409 with the required image, leaving queued investigations untouched

PostgreSQL stores complete review checkpoints in `LiteLLM_LensReview`, alongside Lens records and run history. Schema migrations preserve saved investigations, findings, history, worker credentials and billing assignments without rewriting stored Lens records

Drain active scans, stop workers, back up the database, and deploy all gateway replicas as a coordinated replacement or traffic cutover. Keep traffic paused until every gateway replica uses the selected build and its migrations have completed. Mixed gateway versions sharing Lens data are not supported because every replica must understand the stored records. Pausing workers alone does not prevent dashboard or API writes. Recreate workers with the matching image and their existing tokens, then resume traffic and schedules. Scanning waits until a compatible worker connects

Use the shipped migration history for databases containing Lens data. The schema guard stops `--use_prisma_db_push` before changes if it detects Lens tables that require renaming; run the shipped rename migration against the configured schema before retrying. Fresh databases and databases with the current table names can use database push

A rollback to a gateway that cannot read saved Lens records requires restoring a compatible database backup. Database push from such a build can also remove the review table. Test recovery on a separate database and account for all gateway data written after the backup

The dashboard reads its image from the running gateway. `LENS_WORKER_IMAGE` overrides the registry/image for private deployments. Set an explicit `LENS_WORKER_IMAGE` for worker-only Compose. Verify that the image exists and matches the gateway before deploying it

For source development, use `make lens-dev`, which gives the proxy and source worker the same commit identity. For custom containers, build both from the same checkout with `--build-arg LITELLM_RELEASE_TAG=sha-$(git rev-parse HEAD)` and set the proxy's `LENS_WORKER_IMAGE` to the worker image you built. An unlabelled custom build refuses worker setup and claims instead of guessing from the Python package version. Normal package-index installations use their installed release version

The hourly development pipeline pins all component images to the same selected commit and publishes its chart only after every build and worker smoke test succeeds. The public commit-tagged worker workflow publishes to `ghcr.io/berriai/litellm-lens-worker-dev` on Lens-related changes, so an arbitrary `main` commit may require building your own pair; do not substitute the newest available worker


## Worker dependencies

The worker uses the same digest-pinned Wolfi base and Python version as the component images. Python dependencies and their hashes are locked in `deploy/lens/requirements.lock`. To update them, edit `deploy/lens/requirements.in`, then run `uv pip compile --universal --python-version 3.13 --generate-hashes --no-emit-index-url deploy/lens/requirements.in -o deploy/lens/requirements.lock`. The image installs only the locked wheels with hash verification. CI builds and scans both native architectures

## Python analysis boundary

The `python` tool runs ordinary CPython with the standard library in a fresh child process inside the existing worker container. It receives the selected evidence as `data` over stdin and has its own temporary working directory. It creates no additional container or service. Read and search tools remain available independently of Python

The native worker image builds a syscall policy with libseccomp and includes the full `setpriv` launcher. Each child starts with no inherited worker secrets or open worker files, isolated Python startup, Landlock filesystem restrictions and a default-deny seccomp filter. It can read the Python runtime and its own scratch files. Worker source, installed worker packages, other jobs' files and `/proc` contents are unavailable. Network sockets, child processes, cross-process memory operations, signals to other processes and filesystem metadata mutation are denied, including calls made through `ctypes`. Some metadata inspection, such as `stat`, `access` and `readlink` of known paths, remains possible

Python execution requires a native Linux worker with Landlock ABI 3 or later and seccomp filtering. Build the image for the host architecture. Missing policy files, an incompatible kernel, or an unsupported host such as a macOS source worker returns a clear tool error. There is no unrestricted execution fallback. Keep the container's non-root user, dropped capabilities, no-new-privileges setting, read-only root and writable temporary mount

The worker permits two Python children at once across all investigations. Set `LENS_PYTHON_CONCURRENCY` to a positive integer to change this worker-wide pool. Queued calls consume no child process or scratch directory; cancelling a queued call does not start it. Model, read and search concurrency are separate

| Per-call resource | Default |
| --- | --- |
| Elapsed execution time | 60 seconds |
| CPU time | 30 seconds |
| Process address space | 512 MiB |
| Captured stdout or stderr | 8 MiB per stream |
| Individual scratch file size | 16 MiB |
| Monitored scratch storage | 64 MiB |
| Monitored scratch entries | 2,048 |
| Scratch directory depth | 128 |
| Open file descriptors | 64 |

Evidence is streamed from gateway pages into the confined child without building another complete selection in worker memory. The child decodes the selected data under its memory limit before running the code. The execution wall clock starts after input delivery; gateway fetches keep their HTTP timeouts and remain cancellable. CPU, address-space and file-size limits apply during input decoding as well as computation. Scratch usage is monitored every 50 milliseconds, so a call can temporarily overshoot its scratch allowance. The worker's shared temporary mount supplies the hard aggregate storage ceiling, 1 GiB by default. Accounting includes unlinked open files and files retained only by memory mappings. A mapped scratch inode without an open descriptor or directory entry is conservatively charged at the individual file-size limit, which may overcount small files. Cancellation and limit failures kill and reap the child before removing its scratch directory

Results include `stdout`, `stderr`, `exit_code`, `error` and `output_complete`. Nonzero interpreter exits, confinement failures and resource failures set `error` and `output_complete=false`. Available traceback output is retained. An output-size failure delivers no partial stdout/stderr; the agent can narrow its computation and retry. A successful result retains all captured output without truncation

This is a process boundary sharing the worker's Linux kernel. The checked-in smoke test verifies useful Python operations, filesystem and process restrictions, raw syscall attempts, resource failures, mapping accounting, cleanup and cancellation in the actual image. Run it on the deployment's native architecture and kernel:

```bash
docker build --build-arg LITELLM_RELEASE_TAG=lens-python-test \
  -f deploy/lens/Dockerfile -t lens-worker:python-test .
docker run --rm --pull never --read-only --cap-drop ALL \
  --security-opt no-new-privileges --network none \
  --tmpfs /tmp:rw,noexec,nosuid,size=1g --entrypoint python -i \
  lens-worker:python-test - < tests/proxy_behavior/lens/worker_python_smoke.py
```

The same checks can run through pytest by setting `LENS_TEST_WORKER_IMAGE` to an already-built native image. The worker image CI runs the standalone smoke without adding pytest to the production image
