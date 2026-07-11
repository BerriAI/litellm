# Agent guide: writing e2e tests under `tests/e2e/`

This directory holds live end-to-end suites that prove product correctness against a real running proxy and real provider APIs. Read `CONTRIBUTING.md` for setup and the lifecycle contract, and `CLAUDE.md` for harness layering and coverage-registry grammar. This file is the agent-facing checklist for writing a complete, production-quality e2e test.

## Hard rule: never skip

**Tests under `tests/e2e/` must NEVER skip. Hard failures only.**

Do not call `pytest.skip`, and do not write fixtures that skip when the proxy is down, when provider keys are missing, when Langfuse/Datadog/etc. credentials are missing, or when a model is unavailable. Those conditions are environment or product failures: use `assert`, `pytest.fail`, or raise.

Consequences:

- Missing proxy, missing `OPENAI_API_KEY` / `GEMINI_API_KEY` / etc., missing integration credentials, missing models, network errors on liveness → **hard fail**
- A call the product is expected to complete that returns non-2xx → **hard fail** (see `require_successful_call`)
- Never paper over a broken env with a green skip count

### Shared liveness gate

`tests/e2e/conftest.py` hard-fails every `@pytest.mark.e2e` test when `/health/liveliness` does not answer (`pytest.fail`, never skip). Suite-local fixtures (for example logging credentials) must hard-fail the same way.

Older suites may still contain `pytest.skip` for missing env vars. Treat those as debt. New tests and any suite you edit must hard-fail instead. Do not reintroduce skip-on-missing-proxy or skip-on-missing-creds.

## What a complete feature e2e must do

A feature test is complete only when it walks the feature end to end, in this order:

1. **CREATE** the resource (key / team / budget / model / ...) and immediately queue its deletion via `resources.defer(...)` or `resources.key(...)`.
2. **CONFIGURE** the feature's setting on that resource (attach the callback, assign the budget, set the limit, register custom pricing).
3. **ACT** with real traffic like production: real auth headers, a real model or a deployment you just created, enough calls to actually trigger the behavior.
4. **SETTLE** by polling to a deadline until the side effect lands. Spend flushes on `proxy_batch_write_at` (~60s). Logging backends and metrics are eventually consistent. Use `time.monotonic()` + `POLL_TIMEOUT` / `gateway.poll_timeout` loops with `POLL_INTERVAL` sleeps. **Never sleep once and assert.**
5. **ASSERT recorded state** the feature promises (spend row cost, Langfuse trace cost, Prometheus series, S3 object, ...).
6. **ASSERT enforced behavior** the gateway returns (2xx success, 429 `budget_exceeded`, provider failure status, stream content-type, refusal body, ...).
7. **TEARDOWN** via the `resources` fixture (LIFO, best-effort). Every create path must register a delete.

If a step is missing, the test is not done.

### Both sides of the product promise

You must assert **both** recorded state (step 5) **and** enforced gateway behavior (step 6). A test that only checks "the call went through", or only checks a log without checking the gateway response the customer sees, is incomplete.

For pure logging delivery, step 6 is the gateway outcome the product guarantees on that path (success status + stream shape, or a real provider failure status), and step 5 is the external system (Langfuse trace, Datadog metric, Prometheus series) holding the promised fields.

## Reference implementations

Copy structure from these, not from unit tests:

- Lifecycle + custom pricing dual assert: `llm_translation/test_custom_pricing_e2e.py`
- Logging integration delivery: `logging/test_langfuse_e2e.py`
- Shared gateway + poll helpers: `e2e_gateway.py`
- HTTP transport (only place that may call `requests.*`): `e2e_http.py`
- Resource lifecycle: `lifecycle.py`

## Suite layout

Each subdirectory under `tests/e2e/` is one suite. See `CLAUDE.md` for the folder map. A suite typically has:

- `conftest.py` — suite `client` fixture (and any required credential fixtures that **hard-fail** when unset)
- `*_client.py` — frozen dataclass holding `Gateway` plus suite-specific routes
- `test_*_e2e.py` — feature classes with `@pytest.mark.e2e` and `@pytest.mark.covers(...)`

## Transport and typing rules

1. **Never import or call `requests.*` outside `e2e_http.py`.** CI enforces this via `tests/code_coverage_tests/check_e2e_no_raw_requests.py`. External APIs (Langfuse, etc.) also go through `e2e_http.get` / `post` / `send`.
2. Use the suite `client` fixture and the shared `Gateway`. Do not hand-roll base URLs or master-key headers in the test body.
3. Request and response bodies are **typed pydantic models**. Only model the fields you read. No `Any`, no raw `dict` bodies threaded through tests.
4. Outcomes are a `Result[R]` tagged union. Prefer `match`, or `unwrap(...)` when non-success must fail the test.
5. For status-only LLM outcomes (failures, streams, passthrough), use `StreamingResponse` via `transport.send` / `gateway.chat_stream` and assert on `ok`, `status_code`, `call_id`, `is_streaming`, `chunks`.
6. Prefer public helpers on the suite client. Do not import private `_foo` helpers from client modules into tests; export a public name if the test needs it.
7. When an endpoint returns JSON (for example `POST /team/{id}/callback` → `{"status": "success", "data": ...}`), give it a real response model. `NoBody` is for empty or ignore-shaped teardown routes, not for product responses you care about.

## Markers and coverage registry

- Mark live tests with `@pytest.mark.e2e` (module-level `pytestmark` or class).
- Declare coverage with `@pytest.mark.covers("registry.cell.id", exercised_on=[...])`.
- Cell ids must exist in `coverage_registry/*.yaml`. Logging cells live in `coverage_registry/logging.yaml` and follow `logging.<integration>.<event>.<assertion>`.
- Run `python -m coverage_registry.collector --strict` when you need CI to reject unknown marker ids.
- Prefer one focused test per registry cell behavior. Expand `exercised_on` only when the test actually hits those surfaces.

## Concurrent-run and side-effect safety

The proxy is long-lived and shared. Tests run (and may run in parallel) against the same instance.

- Name every created resource with `unique_marker()` from `e2e_config` (teams, keys, model names, prompts, tags, customer ids).
- Prefer **team-scoped** or **key-scoped** configuration over mutating global proxy callback lists, so concurrent suites do not stomp each other.
- Create deployments via `/model/new` and delete them on teardown when the test needs a private backend (bad keys, custom pricing, isolated models). Do not rely on leftover state from another suite.
- Queue teardown immediately after each create: `resources.defer(lambda: client.delete_...(id))`.

## Failure-path tests

A failure-path e2e must induce a **real product failure**, not a proxy validation skip that never reaches logging or spend.

Good: register a deployment with an invalid upstream API key and call it, so the provider rejects the request after the proxy has accepted it.

Bad: call a model name that does not exist and only assert a 400 from request validation, then claim "failure logging works."

Assert:

- Gateway behavior: non-success status (and that it is the failure class you intended).
- Correlation: `x-litellm-call-id` (or equivalent) is present when the product promises it.
- Recorded state: the integration still received the event (trace / log / metric), with the fields the product promises on failure (for example spend present as `0` or partial).

## Stream-path tests

If the registry cell or product promise involves streaming:

- Call with `stream=True` through `gateway.chat_stream` / the suite helper that uses `transport.stream`.
- Assert `outcome.is_streaming` (`text/event-stream`) and `outcome.chunks > 0`.
- Then settle and assert recorded state (aggregated tokens / spend), not only that the stream opened.

## Spend and cost assertions

Spend assertions must be meaningful enough that a regression which stops logging cost fails the test.

- Prefer `cost is not None and cost > 0` on success paths (or exact rate math when testing custom pricing).
- On failure paths, require the cost field to be present even if the value is `0`.
- When rates are under test, assert input and output components separately against token counts; a total-only check can hide swapped rates.
- Poll until the row or external trace has the cost field you need; do not accept "trace exists" alone when the cell says `logs_spend`.

## Polling template

```python
deadline = time.monotonic() + gateway.poll_timeout
last = None
while time.monotonic() < deadline:
    last = fetch()
    if ready(last):
        return last
    time.sleep(gateway.poll_interval)
pytest.fail("side effect never landed before the deadline")
```

Reuse `Gateway.poll_logs_for_key` / `poll_logs_for_request_id` when the side effect is `/spend/logs`. Suite clients may add analogous poll helpers for external systems.

## Credentials and secrets

- Read keys from the environment (repo root `.env` / `tests/e2e/.env` as documented in `CONTRIBUTING.md`).
- Never print secrets, never commit them, never put them in assertion messages.
- Fixture that needs an integration: load env vars and `pytest.fail(...)` with a clear message listing the required variable names when any are missing.

## Style checklist before you finish

- Feature cases live in a class (`TestLangfuseSpendLogging`, `TestCustomPricing`, ...) so the file reads as a production contract.
- Fully typed; no `Any`; line length ≤ 120.
- No new comments unless essential; existing comments may stay.
- Composition over inheritance; frozen client dataclasses; early returns.
- Do not mutate global proxy config when a team/key/model-scoped knob exists.
- Run the suite against a live proxy before claiming done; proof is curl/logs of real behavior, not a mocked unit test.

## Minimal skeleton

```python
pytestmark = pytest.mark.e2e

class TestFeature:
    @pytest.mark.covers("module.feature.variant.assertion", exercised_on=["chat_completions"])
    def test_feature_behavior(
        self, client: SuiteClient, resources: ResourceManager
    ) -> None:
        marker = unique_marker()
        resource_id = client.create_...(f"e2e-{marker}")
        resources.defer(lambda: client.delete_...(resource_id))

        client.configure_feature(resource_id, ...)

        outcome = client.act(...)
        require_successful_call(outcome)  # or assert the expected failure

        recorded = client.poll_until_...(resource_id, outcome.call_id)
        assert recorded_state_holds(recorded)
        assert gateway_enforced_behavior(outcome)
```

If any of CREATE, CONFIGURE, ACT, SETTLE, dual ASSERT, or TEARDOWN is absent, keep writing until it is not.
