# e2e harness conventions

Code-style rules for writing tests under `tests/e2e/`. The harness already encodes the plumbing; your job is the feature-specific behavior, not reinventing it. For what a complete test must do (the lifecycle contract, asserting both recorded state and enforced behavior) and how to run a suite, see `CONTRIBUTING.md` in this directory. Repo-wide conventions live in the root `CLAUDE.md`

## Suite folders

Each subdirectory under `tests/e2e/` is one suite, scoped to an endpoint family or behavior area. If you add a new folder, you must add a line here describing what kind of tests belong in it, so the layout stays self-describing. `gateway/` is the exception: it holds proxy configuration only and never tests

- `llm_translation/` - LLM endpoint and provider-translation behavior: passthrough, custom pricing, OCR
- `embeddings/` - the `/embeddings` endpoint across providers
- `batches/` - the `/batches` endpoint (placeholder until the first test lands)
- `realtime/` - realtime websocket sessions, including the pipecat audio path
- `budgets/` - budget definition, enforcement, and reset windows (key, team, tag, soft, multi-window)
- `spend_tracking/` - spend logging and cost attribution on `/spend/*`
- `models_mgmt/` - model-management routes (add/update, tpm persistence)
- `logging/` - logging-integration delivery (datadog and friends)
- `security/` - secret handling and log-leak protection
- `router/` - routing and reliability behavior (rate limits, fallbacks, cooldowns)
- `gateway/` - proxy configuration only (`litellm-config.yml`); no tests

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

Request and response bodies are typed pydantic models in `models.py`; only the fields a test reads are modelled, and nothing passes raw dicts. Outcomes come back as a `Result[R]` tagged union (`Success`, `NetworkError`, `UnauthorizedError`, `RateLimitedError`, `ValidationError`, `UnknownApiError`). Handle them with `match`, or call `unwrap(...)` when a non-success should fail the test. The skip-vs-fail split is deliberate: a test marked `e2e` skips when no proxy answers its liveness probe, but once a request reaches the proxy any wrong behavior is a hard failure, never a skip

Mark live tests with `@pytest.mark.e2e` (on the class or the module). Pure coverage of the harness itself carries no marker and runs regardless. Use `scoped_key` for a fresh all-models key that auto-deletes, `resources` when you need to create and tear down more than a key, and `unique_marker()` from `e2e_config` to keep prompts, tags, and customer ids from colliding across concurrent runs and the shared response cache

## Typing

The harness is fully typed and new code must not add `Any` or widen the basedpyright budgets. When a response field is untyped, model it in `models.py` (just the fields you read) and let pydantic validate it, rather than threading a `dict` or `Any` through the test

## Coverage registry

The set of tests we want is a registry checked into this repo, one row per behavior; that file is the definition of done and the denominator. Each e2e test declares what it covers with `@pytest.mark.covers("...")`, and a small collector diffs the registry against the tests and ships coverage to the existing Grafana. No Allure, no new dependencies

Coverage is organized as module > feature > test. There are six modules: LLMs, MCPs, Management/UI, Reliability & Performance, Logging & Guardrails, and Other. A feature is either an endpoint (`/chat/completions`) or a behavior (fallbacks, rate limits; config-driven, with no route of its own). A cell reads like `llm.chat_completions.bedrock_converse.tool_use.stream.works`

The metric is coverage: the share of registry rows that have a passing covering test, reported to Grafana per module so a gap surfaces as an uncovered row rather than a silent absence

### Naming grammar per module

LLMs - endpoint features (subject = the route), seeded from the Claude Code compat matrix

```
llm.<endpoint>.<route>.<capability>.<streaming>.<assertion>
  endpoint   : chat_completions | messages | responses | embeddings | batches | files
               | rerank | images_generations | audio_speech | audio_transcriptions | moderations
  route      : openai | azure_openai | anthropic | bedrock_invoke | bedrock_converse | vertex | azure_foundry
               (vocab varies per endpoint; messages is anthropic-format only)
  capability : basic | tool_use | prompt_cache_5m | prompt_cache_1h | vision | thinking
               | thinking_tool_use | pdf_input | web_search | structured_output | count_tokens
               | tool_search | long_context_1m
  streaming  : stream | nonstream   (omit where n/a)
  assertion  : works | cost_logged
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
  behavior  : fallback | retry | cooldown | timeout | ratelimit | routing | cache | circuit_breaker | perf
  variant   : <trigger>   5xx | context_window | content_policy | 429 | timeout
              <strategy>  simple_shuffle | usage_based | latency_based | cost_based | least_busy
              <dimension> latency | throughput   (perf only; SLO/threshold assertion, not binary)
  assertion : routes_to_fallback | succeeds_within_retries | picks_under_tpm | returns_cached
              | trips_then_recovers | under_slo
  e.g.  reliability.fallback.context_window.routes_to_fallback     exercised_on=[chat_completions]
        reliability.ratelimit.rpm.blocks_over_limit                exercised_on=[chat_completions, messages]
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
