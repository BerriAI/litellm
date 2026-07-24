# Contributors Guide

This directory holds the live end-to-end suites that prove product correctness against a real running proxy and real provider APIs. The goal of this guide is simple: when you ship a feature, you add e2e coverage that walks that feature the way production does, across every route and edge case it touches, so a later change that breaks it fails here first

Read this before adding a test and i recommend reading through CLAUDE.md

When contributing to this directory, please first discuss the change you wish to make via issue or pull request. We require screenshots and proof of your tests working on a live proxy. 


## Setup

The suites run against a live proxy, so bring one up first by running the litellm proxy locally. Point it at a config that prewires the example models the suites use (`gpt-5.5`, `claude-haiku-4-5`, `gemini-2.5-flash`, `openai-text-embedding-3-small`) with keys from your `.env`, and enables prompt storage, a redis cache, and the fast budget rescheduler the quota suites rely on. If your test needs another model, a pricing override, or a guardrail declared up front, add it to that config and read it back in the test rather than hardcoding values

## Running the tests locally

1. Create a `.env` file in this directory with the provider keys the example models use, plus the master key and the Postgres/Redis coordinates your config reads back:

   ```bash
   LITELLM_MASTER_KEY="sk-1234"
   DATABASE_URL="postgresql://llmproxy:dbpassword9090@localhost:5432/litellm"
   REDIS_HOST="localhost"
   REDIS_PORT="6379"
   OPENAI_API_KEY="sk-..."
   ANTHROPIC_API_KEY="sk-..."
   GEMINI_API_KEY="..."
   ```

2. Bring up a Postgres and a Redis for the proxy to use. The repo-root `docker-compose.yml` already defines a Postgres on `5432`; a `docker run -p 6379:6379 redis:7` covers Redis. Point `DATABASE_URL` / `REDIS_HOST` / `REDIS_PORT` at whatever you run

3. Start the litellm proxy locally against your config and confirm it is live:

   ```bash
   set -a && source .env && set +a
   litellm --config <your-e2e-config>.yml --port 4000
   curl -fs http://localhost:4000/health/liveliness
   ```

4. Run a suite against it; the harness reads `LITELLM_PROXY_URL` (default `http://localhost:4000`):

   ```bash
   uv run pytest tests/e2e/llm_translation/ -v
   ```

   The browser tests in the `management/` suite drive the dashboard the proxy serves at `/ui` through playwright, an optional dependency behind `importorskip` (the suite's API tests run without it). It lives in the `e2e-dev` dependency group; install it along with its browser:

   ```bash
   uv sync --inexact --group e2e-dev
   uv run playwright install chromium
   ```

   They also need a proxy whose bundled UI contains the change under test, so run the proxy from your branch (an editable install serves the UI your checkout builds)

Some suites need extra services the bare proxy does not start. The `logging/` OTEL trace-completeness tests read spans back from a jaeger query API at `http://localhost:16686` (override with `E2E_OTEL_QUERY_URL`); run a `jaegertracing/all-in-one` and point `PHOENIX_COLLECTOR_HTTP_ENDPOINT` at its OTLP ingest. The `mcp/` suite needs the deterministic upstream MCP server in `mcp_tests/mcp_e2e_upstream_server.py` reachable by the proxy

Tests marked `@pytest.mark.e2e` hard-fail when no proxy answers `/health/liveliness`, so a run that goes red with `No live proxy` at setup means the proxy isn't up; they never skip for a missing proxy, so an absent proxy can't be mistaken for a pass

## What a complete test looks like

A feature test is complete only when it walks the feature end to end, in this order

1. CREATE the resource (key / team / budget / ...) and immediately queue its deletion
2. CONFIGURE the feature's setting on it (assign the budget, turn on compression, set the limit)
3. ACT; drive real traffic through the gateway exactly like prod does (right model, real auth headers, enough calls to actually trigger the behavior)
4. SETTLE; poll the DB / spend logs until the write lands. Writes are eventually consistent (spend flushes on proxy_batch_write_at, ~60s), so poll to a deadline. Never sleep once
5. ASSERT the recorded state the feature promises (spend > budget, cost reduced, tag attributed, ...)
6. ASSERT the enforced behavior the gateway returns (429 budget_exceeded, block, refusal, ...)
7. TEARDOWN; every resource you created is deleted

### The one rule that makes it complete

It must assert BOTH sides: the recorded state (step 5) AND the enforced behavior (step 6)

A test that only checks "the call went through", or only checks spend without checking the 429, is not complete; it is checking plumbing, not the product promise

### Example: budget enforcement

```
create a key                     -> (1)
assign a budget                  -> (2)
send a bunch of calls            -> (3)
poll for db spend                -> (4)
assert spend > budget            -> (5)
assert status_code == 429        -> (6)  ("budget_exceeded")
key auto-deleted on teardown     -> (7)
```

### The skeleton every test fills in

```
setup     ->  create the resource + queue cleanup
configure ->  apply the feature's knob
act       ->  send real calls like production
settle    ->  poll the DB until the write lands
assert    ->  recorded state is correct   (the feature happened)
assert    ->  gateway enforced it         (the product promise held)
teardown  ->  delete everything you created
```

If a step is missing, the test is not done. That is the whole pattern

## Style: lay the pattern down in a class

Keep the cases for one feature inside a class so the file reads as a spec for how that feature behaves in production. The class name says what is under test; each method is one behavior. Think of it as documenting the contract, with the rough intent being

```python
# pseudo-code to convey intent
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

## Pre-commit steps

Before you push

1. Run `make lint-e2e-basedpyright` (or `make pre-commit` with your changes staged); the harness is fully typed and the gate allows zero basedpyright errors, enforced in CI on any PR touching `tests/e2e/**/*.py`

2. Add the models your test needs to the config your local proxy loads

3. Start the litellm proxy locally and run your suite against it:

   ```bash
   litellm --config <your-e2e-config>.yml --port 4000
   uv run pytest tests/e2e/<your_suite>/ -v
   ```

4. Capture screenshots of the test run and attach them to the PR as proof

5. If a test fails because it surfaced a real issue in the product, flag that explicitly in the PR rather than reworking the test until it passes
