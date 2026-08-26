# e2e harness conventions

Code-style rules for writing tests under `tests/e2e/`. The harness already encodes the plumbing; your job is the feature-specific behavior, not reinventing it. For what a complete test must do (the lifecycle contract, asserting both recorded state and enforced behavior) and how to run a suite, see `CONTRIBUTING.md` in this directory. Repo-wide conventions live in the root `CLAUDE.md`

## Suite folders

Each subdirectory under `tests/e2e/` is one suite, scoped to an endpoint family or behavior area. If you add a new folder, you must add a line here describing what kind of tests belong in it, so the layout stays self-describing. `gateway/` is the exception: it holds proxy configuration only and never tests

- `llm_translation/` - LLM endpoint and provider-translation behavior: passthrough, custom pricing, OCR, and the non-chat inference endpoints (`/v1/responses`, `/v1/messages`, `/embeddings`, `/v1/rerank`, `/v1/audio/speech`, `/v1/images/generations`), each against a deployment the test creates via `/model/new` and deletes on teardown
- `access_control/` - the gateway's authorization and error-shape contract: per-key model allow-lists, route-group permissions (`allowed_routes`), and unknown-model validation
- `embeddings/` - the `/embeddings` endpoint across providers
- `batches/` - the `/batches` endpoint (placeholder until the first test lands)
- `realtime/` - realtime websocket sessions, including the pipecat audio path
- `quota_management/` - quota enforcement and accounting, one subfolder per behavior: `ratelimit/` (rpm/tpm blocks, window reset, pacing headers on live traffic), `budgets/` (budget definition, enforcement, and reset windows: key, team, tag, soft, multi-window), and `spend_tracking/` (spend logging and cost attribution on `/spend/*`)
- `management/` - key/team/user/organization management routes: create/update/delete persistence via the info routes, team membership, and llm-only-key route denials (API surface; not Playwright)
- `a2a/` - the A2A (agent-to-agent) surface: admin registration via `/v1/agents`, proxy-fronted card discovery at `/.well-known/agent-card.json`, and JSON-RPC `message/send` invocation, driving agents backed by the litellm completion bridge (a real provider) and asserting protocol-version normalization (0.3 vs 1.0)
- `mcp/` - the MCP server surface over api_key auth against the real Datadog remote MCP server (see "MCP suite: real Datadog only" below); plus the gateway-managed OAuth (authorization_code) path exercised through `/chat/completions`, the one behavior Datadog's static-header auth cannot reach, seeding the per-user upstream token via the interactive authorize dance driven with the mcp SDK's own OAuth client (headless-browser consent from a saved session) and asserting the completion lists and executes the server's tools with the stored per-user token
- `logging/` - logging-integration delivery (datadog and friends)
- `security/` - secret handling and log-leak protection
- `router/` - routing and reliability behavior (fallbacks, cooldowns)
- `load/` - performance-category tests, kept OUT of the main suite: throughput/load SLO tests are a different testing category from functional e2e (variance-driven, historically flaky) and live outside this suite until re-implemented as their own pipeline (LIT-5163); do not add a live load test that runs in the default collection. What remains here: the weekly session-anomaly test (`test_weekly_session_anomaly_e2e.py`, Claude Code-shaped multi-turn sessions against real providers with ceilings on error rate, cache read/write, turn time, and spend; marked `weekly` and deselected unless `E2E_WEEKLY_ANOMALY` is set, driven by `.github/workflows/weekly_load_anomaly.yml`) and markerless harness unit tests for the Locust/session-anomaly aggregation logic
- `other/` - the holding-pen suite for the `other.*` registry cluster with no home of its own yet: the master-key auth gate and the process-lifecycle health probes (liveness, public readiness, authenticated readiness diagnostics). Promote a cluster out once it is large/stable enough for its own suite
- `gateway/` - proxy configuration only (`litellm-config.yml`); no tests
- `claude_code/` - the Claude Code compatibility matrix: drives the real `claude` CLI (and HTTP probes) against a proxy for each feature x provider cell, reporting tagged-union outcomes via the `compat_result` fixture; ships its own driver/builder/publisher plus `_*_unit_tests/` trees. The HTTP probes ride the shared transport (`ProxyClient.count_tokens` / `ProxyClient.messages`); the CLI-driving path stays bespoke
- `ui/` - the Admin UI browser suite: Playwright in TypeScript, driving the dashboard served by a live proxy on port 4000 (seeded postgres + mock LLM upstream; see its `run_e2e.sh`). It is a self-contained npm package with its own lockfile and does not use the Python harness, pytest markers, or the shared transport; the Python rules in this file (typed models, `Result` unions, basedpyright zero-error gate) do not apply inside it. Its only Python file, `fixtures/mock_llm_server/server.py`, is excluded from the e2e basedpyright gate via the root `pyrightconfig.json`

## MCP suite: real Datadog only

Every test under `tests/e2e/mcp/` must exercise the proxy against the real Datadog remote MCP server. Do not add a compose service, FastMCP fixture, mock upstream, or any other fake MCP host for this suite

- Register via `register_datadog_mcp` in `tests/e2e/mcp/datadog_mcp.py` (or extend that helper if you need a different `toolsets=` / `allowed_tools` slice of the same Datadog endpoint). That posts `/v1/mcp/server` with `url=datadog_mcp_url(...)` and static headers `DD-API-KEY` / `DD-APPLICATION-KEY` from the process env
- Auth is Datadog's documented CI/header path, not a browser OAuth authorize/token dance. Hard-fail when `DD_API_KEY` or `DD_APP_KEY` is missing (`assert_dd_mcp_creds`); never skip for a missing fake upstream
- Prefer calling real Datadog tools that prove the product path (e.g. `search_datadog_logs` for list/call and permission denials). Seed a unique marker (`e2e-datadog-mcp-*`) in a chat completion when you need a log the tool can find; dual-read with `dd_logs` from conftest when delivery matters
- Delete the MCP server (and any keys) through `resources.defer` the same way every other suite tears down
- If a new MCP behavior cannot be covered with Datadog's tool surface, say so in the PR and get agreement before inventing another upstream; the default is always Datadog
- The one standing exception is `test_mcp_chat_completion_oauth_e2e.py`. Datadog authenticates with the static `DD-API-KEY` / `DD-APPLICATION-KEY` headers and exposes no authorize/token dance at all, so it cannot exercise gateway-managed OAuth or per-user token seeding in any form. That test drives a real Linear MCP server instead; it is still a real remote upstream, so the no-mock, no-fixture rule above holds unchanged

## Lay the pattern down in a class

Keep the cases for one feature inside a class so the file reads as a spec for how that feature behaves in production. The class name says what is under test; each method is one behavior. Think of it as documenting the contract, with the rough intent being

```python
# pseudo-code to convey intent, not the real API
class TestPromptCompression:
    def test_prompt_compression_add_to_virtual_key(self):
        new_key = self.resources.create_key(user_id, compression=True)  # turn the feature on
        resources._defer(new_key)  # queue key deletion

    def test_prompt_compression_accumulate_spend(self, key_id, user_id):
        for _ in range(10):
            response = self.resources.proxy.post("gemini-2.5-flash", key_id, user_id)
        compressed_value = ...
        assert response.cost == compressed_value  # the cost was actually reduced
```

That snippet only conveys intent. What you actually write uses the real harness: the `client` fixture for your suite, the `scoped_key` fixture for an auto-deleted key, typed pydantic bodies from `models.py`, and `unwrap(...)` on the tagged-union result. `tests/e2e/llm_translation/test_custom_pricing_e2e.py` is the reference to copy from; it creates a scoped key, drives a real gemini call, polls `/spend/logs` to a deadline for the cost-breakdown row, then asserts the input and output costs match the configured custom rates and that a sibling deployment kept its own price. Read it before writing yours

## Use the shared transport; never touch requests directly

Every HTTP call goes through the shared transport, never through `requests.*` in a test. `e2e_http.py` is the only module permitted to call `requests.*`, and that is enforced in CI by `tests/code_coverage_tests/check_e2e_no_raw_requests.py`. A test that imports requests will fail the check

The shape is layered so tests stay declarative

`transport.py` exposes a `Transport` Protocol with `post`, `get`, `delete`, `send`, `stream`, `probe`, plus `bearer(key)` and the `master` header. `HttpTransport` fulfils it, and `SplitTransport` routes each call by path to the data plane or the control plane so a split control-plane/data-plane deployment works without any change in the test

`proxy_client.py` holds `ProxyClient`, a frozen dataclass that wraps a `Transport` and adds the operations tests reuse: `generate_key` / `delete_key` / `key_info`, `model_info`, the LLM calls `chat` / `chat_stream` / `embed` / `ocr`, the spend read-back `spend_logs`, and the poll helpers `poll_logs_for_key` / `poll_logs_for_request_id` that loop to `poll_timeout` instead of sleeping once. It is exposed as the session-scoped `proxy` fixture (see tests/e2e/conftest.py), which each suite's `client` fixture depends on and injects. Add a new route as a method here so other suites get it for free

Each suite provides its own `client` fixture (see `llm_translation/passthrough_client.py`), a frozen dataclass that holds the shared `ProxyClient` (as `.proxy`) and adds suite-specific routes. Cleanup runs through that same `ProxyClient`, so whatever keys or customers your test creates get torn down by the `resources` fixture

Request and response bodies are typed pydantic models in `models.py`; only the fields a test reads are modelled, and nothing passes raw dicts. Outcomes come back as a `Result[R]` tagged union (`Success`, `NetworkError`, `UnauthorizedError`, `RateLimitedError`, `ValidationError`, `UnknownApiError`). Handle them with `match`, or call `unwrap(...)` when a non-success should fail the test. The harness hard-fails and never skips: a test marked `e2e` fails when no proxy answers its liveness probe, and once a request reaches the proxy any wrong behavior is likewise a hard failure, so a missing proxy turns the run red instead of being mistaken for a pass

Mark live tests with `@pytest.mark.e2e` (on the class or the module). Pure coverage of the harness itself carries no marker and runs regardless. Use `scoped_key` for a fresh all-models key that auto-deletes, `resources` when you need to create and tear down more than a key, and `unique_marker()` from `e2e_config` to keep prompts, tags, and customer ids from colliding across concurrent runs and the shared response cache

## Record and replay fixtures

`E2E_FIXTURE_MODE` scopes the proxy's provider-bound traffic: `live` (the default, and what an unset variable means: nothing changes), `record` (the proxy's provider calls are forwarded to the real provider through a local edge server and written to a fixture bundle), or `replay` (the edge answers those calls from the bundle, so the run makes zero provider calls and spends nothing). Test-to-proxy traffic always goes over the wire in every mode: record and replay both need the live proxy and database, because the point is that key auth, routing, cost calculation, and spend-log writes execute for real while only the provider is swapped out. Breaking any of those in the proxy turns a replay run red

The seam is `provider_edge.py`: `start_provider_edge` boots an in-process HTTP server (one shared instance per pytest process, `e2e_config.provider_edge_base` is the accessor) that mounts each supported provider under a path prefix (`EDGE_MOUNTS`: `/openai` -> `https://api.openai.com`, `/anthropic` -> `https://api.anthropic.com`). A test participates by registering its deployment with `api_base=provider_edge_base("openai")` plus the provider's path suffix; `quota_management/spend_tracking/test_provider_edge_spend_e2e.py` is the reference. In live mode the accessor returns None and the deployment defaults to the real provider, so an edge-wired test runs in all three modes unchanged. Non-wired tests hit their providers live in every mode. The edge binds `E2E_PROVIDER_EDGE_BIND_HOST` (default 127.0.0.1) and advertises `E2E_PROVIDER_EDGE_ADVERTISE_HOST` in the api_base it hands out, for proxies running in containers

A bundle (default `tests/e2e/.fixtures`, override with `E2E_FIXTURE_DIR`) is a directory: `manifest.json` carries the record timestamp, harness git version, and format version, and each test gets a subdirectory holding one JSON file per provider call in call order (`0000-post-openai-v1-chat-completions.json`). Request headers are never stored (provider credentials never touch disk), non-JSON request bodies store a canonicalized sha256 digest instead of the bytes, `multipart/form-data` bodies store their ordinary fields plus a JSON list of the uploaded parts' `[field, filename, content-type]` triples and a digest of their content, so the per-request random boundary and the envelope never reach the key, and responses store status, filtered headers, and the verbatim body base64-encoded, which is part of why bundles are gitignored. Responses come in two shapes told apart by a `kind` tag: an ordinary one holding a single base64 body, and, for a response the provider streamed (`content-type: text/event-stream`), one holding its transfer chunks in order plus why the stream ended early if it did, so replay reproduces the split points the provider chose instead of one coalesced body. `fixture_bundle.py` owns the format, and `BUNDLE_FORMAT_VERSION` is checked on load, so a bundle recorded under older rules is refused by name rather than partially read. Record serves the proxy the same filtered stored response replay will serve later, chunk for chunk on a stream, so the two modes are byte-identical from the proxy's side of the socket

Multipart identity is the fiddly corner, and the rules exist because each one had a collision behind it. A part counts as an upload when it carries a filename or declares its own content type, and everything else is an ordinary field. Field names get a `name[n]` suffix on repeats, with a literal `[` doubled first, so a form that repeats `purpose` never keys the same as one that literally sends `purpose[1]`. A field whose name reads as a credential is stored as `<secret>`, which stays key-preserving because the key is recomputed from the stored request rather than saved alongside it, so the live request carrying the real value still matches its redacted fixture. A field value that is not UTF-8 is stored as a base64 sha256 digest, base64 and not hex because the canonicalizer rewrites any 64-character hex run to `<sha256>` and would fold every binary value onto one key. The uploaded parts contribute a JSON list rather than a `field:filename` string, so a separator inside a filename cannot impersonate a field boundary, and their byte length is stored for a reader's benefit but deliberately left out of the key, since the canonicalizer absorbs timestamp and id drift inside a file that changes its length

Replay matches calls per test by canonical key: `fixture_canonical.py` canonicalizes the recorded request (volatile headers and credential fields out, unique markers, generated ids, uuids, and timestamps replaced with fixed placeholders, object keys sorted) and the key is the method, edge path, and a content hash, so identity survives re-records and machine changes while any real content drift comes back as an HTTP 599 naming the computed key, the closest recorded key with its file, and a content diff, and never falls through to a live call. Matching is order-independent across distinct keys (concurrent calls may interleave) and FIFO within one key (a retry loop replays its responses in recorded order); a passed test must also consume its whole recording, or teardown fails it naming a leftover key. Either way the fix is always to re-record with `E2E_FIXTURE_MODE=record`. Every rewrite rule lives in `fixture_canonical.py`, so a new volatile header, credential field name, or generated-id shape is one edit there. Record starts fresh every time: it wipes the previous bundle (refusing to wipe a directory that is not a bundle) and never reads it. A replay bundle whose manifest is older than seven days hard-fails at collection time naming the bundle's age, so replay can never certify against fixtures that have drifted more than a week from the live providers

A replayed response carries the recorded provider response id, and `LiteLLM_SpendLogs.request_id` (the table's primary key) is that id, so a replay against a database that still holds the record run's rows silently dedupes its spend inserts and any spend assertion goes red with zero matching rows and nothing in the proxy log. Run both modes with `E2E_RESET_SPEND_LOGS=1` (plus `DATABASE_URL` in the runner env) so each session truncates the table after itself, or replay against a fresh database, which is the CI shape

The same id reuse reaches the managed-object tables. A replayed `/v1/files` or `/v1/batches` response carries the recorded provider object id, and `LiteLLM_ManagedObjectTable.model_object_id` is unique, so a unified batch create replayed against a database that still holds the record run's row fails on a Prisma unique-constraint violation, which surfaces as a 500, makes the router retry, and exhausts the recording. Replay the batches suite against a fresh database, or truncate `LiteLLM_ManagedObjectTable` and `LiteLLM_ManagedFileTable` before the run

Edge-wired today: `quota_management/spend_tracking/test_provider_edge_spend_e2e.py` (the reference), `llm_translation/test_chat_completions_contract_e2e.py`, the OpenAI registrations in `llm_translation/test_embeddings_endpoint_e2e.py`, the Anthropic deployments in `llm_translation/test_messages_e2e.py` including the streaming test, and the OpenAI batch deployment behind `batches/` (`capabilities.openai_batch_params`). The mount base is not the same for both providers: OpenAI deployments register `f"{base}/v1"`, Anthropic deployments register `base` on its own, because litellm's Anthropic handler appends `/v1/messages` to `api_base` itself where the OpenAI handler appends only `/chat/completions`. Recording one suite locally is two runs against a proxy you already have up:

```bash
E2E_FIXTURE_MODE=record E2E_FIXTURE_DIR=/tmp/e2e-fixtures E2E_RESET_SPEND_LOGS=1 uv run pytest tests/e2e/llm_translation/test_chat_completions_contract_e2e.py
E2E_FIXTURE_MODE=replay E2E_FIXTURE_DIR=/tmp/e2e-fixtures E2E_RESET_SPEND_LOGS=1 uv run pytest tests/e2e/llm_translation/test_chat_completions_contract_e2e.py
```

Point the proxy at bogus provider credentials for the replay run and it still has to pass: that is the whole proof that nothing left the process. Bundles are never committed. `tests/e2e/.fixtures` is gitignored because a bundle holds verbatim provider response bodies and hard-fails after seven days. CI records and replays this lane on a schedule in `.github/workflows/e2e_record_replay.yml`, publishing the bundle as a private `e2e-fixtures-bundle` artifact instead of committing it, selecting the tests with the `@pytest.mark.replayable` marker, and proving the bogus-credentials replay hermetic by counting provider egress with `.github/scripts/e2e_egress_sentinel.py`

Current limits: Bedrock cannot be mounted (SigV4 signs the Host header, so a rewritten api_base fails signature verification), deployments baked into the proxy's config file cannot be edge-wired (only `/model/new` registrations can carry the edge api_base), and a file upload routed by `custom_llm_provider` through the proxy's `files_settings` block never passes a deployment at all, so the batches `model_param` and `provider_fallback` scenarios keep uploading live in every mode

## Typing

The harness is fully typed with no error budget: `make lint-e2e-basedpyright` must report zero basedpyright errors, and CI enforces that on any PR touching `tests/e2e/**/*.py`. When a response field is untyped, model it in `models.py` (just the fields you read) and let pydantic validate it, rather than threading a `dict` or `Any` through the test

## Coverage registry

The set of tests we want is a registry checked into this repo, one row per behavior; that file is the definition of done and the denominator. Each e2e test declares what it covers with `@pytest.mark.covers("...")`, and a small collector diffs the registry against the tests and ships coverage to the existing Grafana. No Allure, no new dependencies

Coverage is organized as module > feature > test. Dashboard modules are `Core LLMs`, `Non-Core LLMs`, `MCPs`, `Management/UI`, `Reliability & Performance`, `Quota Management`, `Logging & Guardrails`, and `Other`. The Loki stdout formatter maps those display modules to log-safe labels (`core_llms`, `non_core_llms`, `mcp`, `management_ui`, `reliability_performance`, `quota_management`, `logging_guardrails`, and `other`) without changing JSON or Prometheus labels. A feature is either an endpoint (`/chat/completions`) or a behavior (fallbacks, rate limits; config-driven, with no route of its own). A cell reads like `llm.chat_completions.bedrock_converse.tool_use.stream.works`

The metric is coverage: the share of registry rows that have a passing covering test, reported to Grafana per module so a gap surfaces as an uncovered row rather than a silent absence

Tests do not declare a dashboard module directly. They only declare the registry cell id with `@pytest.mark.covers("...")`; the registry row decides the module, tier, endpoint, and dashboard rollup. Run `python -m coverage_registry.collector --strict` when you want CI to reject unknown marker ids. Add `--fail-on-collection-errors` when the job should also fail on pytest collection errors.

Skipping a test gives its cell back to the gap list: the collector counts a cell as covered only when a test pytest would actually run declares it, and prints the cells left claimed only by skipped tests. So a `@pytest.mark.skip` on a red cell is honest bookkeeping, not a way to keep the number up.

### Naming grammar per module

LLMs - endpoint features (subject = the route), seeded from the Claude Code compat matrix. `chat_completions`, `messages`, and `responses` roll up to `Core LLMs`. Other LLM endpoints, including `batches` and `realtime`, roll up to `Non-Core LLMs`.

```
llm.<endpoint>.<route>.<capability>.<streaming>.<assertion>
  endpoint   : chat_completions | messages | responses | embeddings | batches | files
               | rerank | images_generations | audio_speech | audio_transcriptions | moderations
               | realtime
  route      : openai | azure_openai | anthropic | bedrock_converse | bedrock_invoke | vertex
               | azure_foundry | cohere | together_ai
               (vocab varies per endpoint; messages is anthropic-format only)
  capability : basic | tool_use | prompt_cache_5m | vision | thinking | structured_output
               | service_tier | mid_conversation_system
  streaming  : stream | nonstream   (omit where n/a)
  assertion  : works | cost_logged | cache_hit
  label (not in id): model = haiku-4.5 | sonnet-4.6 | opus-4.7 | gpt-*
  e.g.  llm.chat_completions.bedrock_converse.tool_use.stream.works
        llm.messages.anthropic.prompt_cache_1h.nonstream.cache_hit
```

Management / UI - endpoint features (surface tag: api | ui)

```
mgmt.<endpoint>.<assertion>
  endpoint  : key.generate | key.update | key.delete | team.new | user.new
              | budget.new | model.add | ... (one per management route)
  assertion : persists | member_forbidden | admin_only | happy_path
  e.g.  mgmt.key.generate.persists        (surface=api)
        mgmt.key.generate.happy_path      (surface=ui)
```

MCPs - endpoint features with the protocol op as the variant

```
mcp.<operation>.<auth_family>.<assertion>
  operation   : list_tools | call_tool | list_resources | read_resource | list_prompts | get_prompt
  auth_family : none | api_key | bearer | oauth
  assertion   : succeeds | denied_without_permission
  e.g.  mcp.call_tool.oauth.succeeds
```

Reliability & Performance - behavior features (no route; endpoint is exercised_on)

```
reliability.<behavior>.<variant>.<assertion>
  behavior  : fallback | retry | cooldown | timeout | routing | cache | circuit_breaker | perf
  variant   : <trigger>   5xx | context_window | content_policy | 429 | timeout
              <strategy>  simple_shuffle | usage_based | latency_based | cost_based | least_busy
              <dimension> latency | throughput | session_anomaly   (perf only; SLO/threshold assertion, not binary)
  assertion : routes_to_fallback | succeeds_within_retries | picks_under_tpm | returns_cached
              | trips_then_recovers | under_slo
  e.g.  reliability.fallback.context_window.routes_to_fallback     exercised_on=[chat_completions]
        reliability.cooldown.429.trips_then_recovers               exercised_on=[chat_completions, messages]
```

Quota Management - behavior features (entity- or config-driven caps and their accounting; endpoint is exercised_on)

```
quota_management.<behavior>.<variant>.<assertion>
  behavior  : ratelimit | budget | spend_tracking
  variant   : <ratelimit>      rpm | tpm | priority_generous | priority_strict
              <budget>         key | internal_user | end_user | organization | team | team_member | tag
                               | model_max | soft | key_multi_window | team_multi_window
                               | fallback | spend_counter
              <spend_tracking> chat_completions | stream | messages_bridge | embeddings
                               | cache_hit | key_rollup | concurrent_burst | tags | end_user
                               | per_model | failure | spend_calculate | pagination
  assertion : blocks_over_limit | resets_after_window | headers_report_remaining | picks_under_tpm
              | blocks_then_resets | resets_windows_independently | alerts_without_blocking
              | isolates_per_model | isolates_per_member | enforced_across_keys | routes_to_fallback
              | reseed_matches_db | logs_cost | zero_cost
              | matches_sum_of_logs | loses_no_spend | attributes_spend | writes_own_rows
              | writes_failure_row | returns_cost | keeps_total
  e.g.  quota_management.ratelimit.rpm.blocks_over_limit           exercised_on=[chat_completions, messages]
        quota_management.budget.key.blocks_over_limit              exercised_on=[chat_completions]
```

Logging & Guardrails - behavior features (config-driven; endpoint is exercised_on)

```
logging.<integration>.<event>.<assertion>
  integration : langfuse | s3 | otel | prometheus | datadog | ...
  event       : success | failure | stream
  assertion   : logs_spend | writes_object | exports_metric
  e.g.  logging.langfuse.success.logs_spend                        exercised_on=[chat_completions]

guardrail.<provider>.<hook_point>.<assertion>
  provider   : presidio | lakera | bedrock | aporia | ...
  hook_point : pre_call | post_call | during | logging_only
  assertion  : blocks | masks | allows
  e.g.  guardrail.presidio.pre_call.masks                          exercised_on=[chat_completions]
```

Other - holding pen (endpoint or behavior)

```
other.<area>.<case>.<assertion>
  area : auth | lifecycle | config | ...
  rule : audited periodically; a cluster here promotes to a new component
  e.g.  other.auth.jwt.valid_token_allows
        other.lifecycle.readiness.reports_db
```

## Hard Rules
- no monkeypatching or mock tests, and never substitute a unit test for e2e feature coverage: a product feature is proven end to end against a live proxy, not with a unit test. if a contributor asks you to write an end to end test, do NOT stage a unit test of the feature with it; if you find a product gap, call it out in the PR description. tests that cover the harness itself are the exception and are allowed (for example `coverage_registry/test_collector.py`, which unit-tests the coverage collector): they carry no `e2e` marker, exercise harness plumbing rather than a product feature, and run whether or not a proxy is up

- use model management endpoints to create new models for a test. this could be in a conftest / inline for each test. ask the user what they want.

- do not overengineer a test, i need you to write readable, clean code of what would look like a natural user scenario

- when it comes to typing an input schema for an api endpoint, have it type X = A | B | C ... where X = exhaustive union of all supported input schemas and A, B, C typically are composed by a base type. types are only pretty for a api request / response body. make sure to compose types instead of repeating the same base attributes over and over again.
 
- spin up a local proxy by running the litellm proxy locally (`litellm --config <your-e2e-config>.yml --port 4000`; see CONTRIBUTING.md), make sure all tests pass. if a test fails due to an internally found issue, let users know to create a linear ticket for it. 

- do not use xfail markers, tests should be written in a form that the end user expects it to pass
