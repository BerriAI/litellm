# e2e harness conventions

Code-style rules for writing tests under `tests/e2e/`. The harness already encodes the plumbing; your job is the feature-specific behavior, not reinventing it. For what a complete test must do (the lifecycle contract, asserting both recorded state and enforced behavior) and how to run a suite, see `CONTRIBUTING.md` in this directory. Repo-wide conventions live in the root `CLAUDE.md`

## Suite folders

Each subdirectory under `tests/e2e/` is one suite, scoped to an endpoint family or behavior area. If you add a new folder, you must add a line here describing what kind of tests belong in it, so the layout stays self-describing. `gateway/` is the exception: it holds proxy configuration only and never tests

- `llm_translation/` - LLM endpoint and provider-translation behavior: passthrough, custom pricing, OCR, realtime, batches, and every LLM data-plane endpoint (`/chat/completions`, `/v1/responses`, `/v1/messages`, `/embeddings`, `/v1/rerank`, `/v1/audio/speech`, `/v1/images/generations`, `/v1/batches`, `/v1/realtime`, etc.), each against a deployment the test creates via `/model/new` and deletes on teardown where applicable. Any test whose primary subject is an LLM endpoint belongs here, even if the endpoint uses websockets, files, provider passthrough, or a non-chat modality.
- `access_control/` - the gateway's authorization and error-shape contract: per-key model allow-lists, route-group permissions (`allowed_routes`), and unknown-model validation
- `budgets/` - budget definition, enforcement, and reset windows (key, team, tag, soft, multi-window)
- `rate_limits/` - rate-limit enforcement behavior across keys, teams, models, tags, and endpoint families; endpoint-specific LLM translation assertions still live under `llm_translation/`
- `spend_tracking/` - spend logging and cost attribution on `/spend/*`
- `management/` - key/team/user/organization management routes: create/update/delete persistence via the info routes, team membership, and llm-only-key route denials; also the dashboard UI behavior on top of them, driven through the proxy-served UI at /ui with playwright (optional dep behind importorskip)
- `mcp/` - MCP protocol behavior and coverage rows: tool/resource/prompt operations, auth-family handling, and MCP-specific error shapes
- `logging/` - logging-integration delivery (datadog and friends)
- `reliability/` - routing and reliability behavior (fallbacks, retries, cooldowns, routing strategies)
- `other/` - temporary holding area for coverage rows that do not yet have a stable suite owner; promote rows out of here when a module boundary becomes clear
- `security/` - secret handling and log-leak protection
- `gateway/` - proxy configuration only (`litellm-config.yml`); no tests

Do not add new top-level folders for LLM data-plane endpoints. Put endpoint-specific
LLM suites under `llm_translation/` and create a subfolder there only when the
endpoint needs its own client, fixtures, or coverage matrix.

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

## Coverage metadata

Every collected pytest under `tests/e2e` must declare the e2e surface it covers with `@pytest.mark.e2e_coverage(...)`. The marker is the source of truth; there are no coverage YAML files. Use a module-level `pytestmark` when all tests in a file cover the same surface, or add per-test markers when a file mixes endpoints/providers.

```python
pytestmark = [
    pytest.mark.e2e,
    pytest.mark.e2e_coverage(
        module="core_llms",
        endpoint="/chat/completions",
        provider="openai",
        params=["tools", "streaming"],
    ),
]
```

Required fields:

- `module`: one of `core_llms`, `non_core_llms`, `access_control`, `budgets`, `spend_tracking`, `management`, `mcp`, `rate_limits`, `reliability`, `logging`, `guardrails`, or `other`.
- `endpoint`: a known endpoint/surface from `coverage_registry/schema.py`, such as `/chat/completions`, `/v1/messages`, `/v1/batches`, `/budget/*`, or `/spend/*`.
- `provider`: a known provider/integration from `coverage_registry/schema.py`, such as `proxy`, `openai`, `anthropic`, `vertex_ai`, or `multiple`.
- `params`: one or more explicit lowercase parameter/behavior names, such as `tools`, `streaming`, `key_rpm_limit`, `budget_enforcement`, or `spend_routes`.

The collector reads these markers with pytest collect-only and reports unique endpoint x provider x parameter units plus test counts by module. Run this before opening or updating a PR:

```bash
cd tests/e2e && PYTHONPATH=. python -m coverage_registry.check_coverage_sync
```

The sync check fails if any collected test is missing `e2e_coverage`, uses an unknown module/endpoint/provider, has empty params, or has pytest collection errors.

### Marker value grammar

Use stable, dashboard-safe values. Do not include spaces or prose in marker fields. If a new endpoint or provider is legitimate, add it to `coverage_registry/schema.py` in the same PR so the CI check teaches the dashboard about it.

- Core LLM endpoint tests: `module="core_llms"`, endpoints `/chat/completions`, `/v1/messages`, or `/v1/responses`.
- Other LLM endpoint tests: `module="non_core_llms"`, endpoints such as `/v1/batches`, `/v1/realtime`, `/v1/embeddings`, `/v1/audio/speech`, `/v1/images/generations`, or `/rerank`.
- Proxy behavior tests: use the owning module (`budgets`, `rate_limits`, `access_control`, `spend_tracking`, `management`) and the route family being exercised.
- Harness/tooling tests: use `module="other"` or `module="reliability"` with endpoint `e2e_harness` or `coverage_registry`.

Prefer precise params like `key_rpm_limit`, `budget_enforcement`, `prompt_cache`, `tool_calling`, or `spend_routes` over generic params like `works`.
