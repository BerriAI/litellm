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
- `management/` - key/team/user/organization management routes: create/update/delete persistence via the info routes, team membership, and llm-only-key route denials; also the dashboard UI behavior on top of them, driven through the proxy-served UI at /ui with playwright (optional dep behind importorskip)
- `mcp/` - the MCP server surface over api_key auth against the real Datadog remote MCP server only (see "MCP suite: real Datadog only" below)
- `logging/` - logging-integration delivery (datadog and friends)
- `security/` - secret handling and log-leak protection
- `router/` - routing and reliability behavior (fallbacks, cooldowns)
- `gateway/` - proxy configuration only (`litellm-config.yml`); no tests
- `claude_code/` - the Claude Code compatibility matrix: drives the real `claude` CLI (and HTTP probes) against a proxy for each feature x provider cell, reporting tagged-union outcomes via the `compat_result` fixture; ships its own driver/builder/publisher plus `_*_unit_tests/` trees, and does not use the shared transport harness

## MCP suite: real Datadog only

Every test under `tests/e2e/mcp/` must exercise the proxy against the real Datadog remote MCP server. Do not add a compose service, FastMCP fixture, mock upstream, or any other fake MCP host for this suite

- Register via `register_datadog_mcp` in `tests/e2e/mcp/datadog_mcp.py` (or extend that helper if you need a different `toolsets=` / `allowed_tools` slice of the same Datadog endpoint). That posts `/v1/mcp/server` with `url=datadog_mcp_url(...)` and static headers `DD-API-KEY` / `DD-APPLICATION-KEY` from the process env
- Auth is Datadog's documented CI/header path, not a browser OAuth authorize/token dance. Hard-fail when `DD_API_KEY` or `DD_APP_KEY` is missing (`assert_dd_mcp_creds`); never skip for a missing fake upstream
- Prefer calling real Datadog tools that prove the product path (e.g. `search_datadog_logs` for list/call and permission denials). Seed a unique marker (`e2e-datadog-mcp-*`) in a chat completion when you need a log the tool can find; dual-read with `dd_logs` from conftest when delivery matters
- Delete the MCP server (and any keys) through `resources.defer` the same way every other suite tears down
- If a new MCP behavior cannot be covered with Datadog's tool surface, say so in the PR and get agreement before inventing another upstream; the default is always Datadog

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
            response = self.resources.gateway.post("gemini-2.5-flash", key_id, user_id)
        compressed_value = ...
        assert response.cost == compressed_value  # the cost was actually reduced
```

That snippet only conveys intent. What you actually write uses the real harness: the `client` fixture for your suite, the `scoped_key` fixture for an auto-deleted key, typed pydantic bodies from `models.py`, and `unwrap(...)` on the tagged-union result. `tests/e2e/llm_translation/test_custom_pricing_e2e.py` is the reference to copy from; it creates a scoped key, drives a real gemini call, polls `/spend/logs` to a deadline for the cost-breakdown row, then asserts the input and output costs match the configured custom rates and that a sibling deployment kept its own price. Read it before writing yours

## Use the shared transport; never touch requests directly

Every HTTP call goes through the shared transport, never through `requests.*` in a test. `e2e_http.py` is the only module permitted to call `requests.*`, and that is enforced in CI by `tests/code_coverage_tests/check_e2e_no_raw_requests.py`. A test that imports requests will fail the check

The shape is layered so tests stay declarative

`transport.py` exposes a `Transport` Protocol with `post`, `get`, `delete`, `send`, `stream`, `probe`, plus `bearer(key)` and the `master` header. `HttpTransport` fulfils it, and `SplitTransport` routes each call by path to the data plane or the control plane so a split control-plane/data-plane deployment works without any change in the test

`e2e_gateway.py` holds `Gateway`, a frozen dataclass that wraps a `Transport` and adds the operations tests reuse: `generate_key` / `delete_key` / `key_info`, `model_info`, the LLM calls `chat` / `chat_stream` / `embed` / `ocr`, the spend read-back `spend_logs`, and the poll helpers `poll_logs_for_key` / `poll_logs_for_request_id` that loop to `poll_timeout` instead of sleeping once. Add a new route as a method here so other suites get it for free

Each suite provides its own `client` fixture (see `llm_translation/passthrough_client.py`), a frozen dataclass that holds the shared `Gateway` and adds suite-specific routes. Cleanup runs through that same `Gateway`, so whatever keys or customers your test creates get torn down by the `resources` fixture

Request and response bodies are typed pydantic models in `models.py`; only the fields a test reads are modelled, and nothing passes raw dicts. Outcomes come back as a `Result[R]` tagged union (`Success`, `NetworkError`, `UnauthorizedError`, `RateLimitedError`, `ValidationError`, `UnknownApiError`). Handle them with `match`, or call `unwrap(...)` when a non-success should fail the test. The harness hard-fails and never skips: a test marked `e2e` fails when no proxy answers its liveness probe, and once a request reaches the proxy any wrong behavior is likewise a hard failure, so a missing proxy turns the run red instead of being mistaken for a pass

Mark live tests with `@pytest.mark.e2e` (on the class or the module). Pure coverage of the harness itself carries no marker and runs regardless. Use `scoped_key` for a fresh all-models key that auto-deletes, `resources` when you need to create and tear down more than a key, and `unique_marker()` from `e2e_config` to keep prompts, tags, and customer ids from colliding across concurrent runs and the shared response cache

## Typing

The harness is fully typed with no error budget: `make lint-e2e-basedpyright` must report zero basedpyright errors, and CI enforces that on any PR touching `tests/e2e/**/*.py`. When a response field is untyped, model it in `models.py` (just the fields you read) and let pydantic validate it, rather than threading a `dict` or `Any` through the test

## Coverage registry

The set of tests we want is a registry checked into this repo, one row per behavior; that file is the definition of done and the denominator. Each e2e test declares what it covers with `@pytest.mark.covers("...")`, and a small collector diffs the registry against the tests and ships coverage to the existing Grafana. No Allure, no new dependencies

Coverage is organized as module > feature > test. Dashboard modules are `Core LLMs`, `Non-Core LLMs`, `MCPs`, `Management/UI`, `Reliability & Performance`, `Quota Management`, `Logging & Guardrails`, and `Other`. The Loki stdout formatter maps those display modules to log-safe labels (`core_llms`, `non_core_llms`, `mcp`, `management_ui`, `reliability_performance`, `quota_management`, `logging_guardrails`, and `other`) without changing JSON or Prometheus labels. A feature is either an endpoint (`/chat/completions`) or a behavior (fallbacks, rate limits; config-driven, with no route of its own). A cell reads like `llm.chat_completions.bedrock_converse.tool_use.stream.works`

The metric is coverage: the share of registry rows that have a passing covering test, reported to Grafana per module so a gap surfaces as an uncovered row rather than a silent absence

Tests do not declare a dashboard module directly. They only declare the registry cell id with `@pytest.mark.covers("...")`; the registry row decides the module, tier, endpoint, and dashboard rollup. Run `python -m coverage_registry.collector --strict` when you want CI to reject unknown marker ids. Add `--fail-on-collection-errors` when the job should also fail on pytest collection errors.

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
              <dimension> latency | throughput   (perf only; SLO/threshold assertion, not binary)
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
              <spend_tracking> chat_completions | stream | embeddings | cache_hit | key_rollup
                               | concurrent_burst | tags | end_user | per_model | failure
                               | spend_calculate | pagination
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
 
- use the docker-compose to your advantage and spin up a local proxy, make sure all tests pass. if a test fails due to an internally found issue, let users know to create a linear ticket for it. 

- do not use xfail markers, tests should be written in a form that the end user expects it to pass
