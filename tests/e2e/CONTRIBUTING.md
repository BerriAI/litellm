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

2. Bring up a Postgres and a Redis for the proxy to use. The repo-root `docker-compose.yml` already defines a Postgres on `5432`; a `docker run -p 6379:6379 redis:7` covers Redis. Point `DATABASE_URL` / `REDIS_HOST` / `REDIS_PORT` at whatever you run. Tests that read Redis directly default to the deployed shape (TLS + cluster mode) whenever `REDIS_HOST` is set, so for a local standalone Redis also set `REDIS_CLUSTER=false` and `REDIS_SSL=false` (plus `REDIS_PASSWORD` when your Redis requires auth)

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

### Record and replay

Record/replay scopes to the proxy's provider-bound traffic only. In `E2E_FIXTURE_MODE=record` the harness boots a local provider-edge server, edge-wired tests register their deployments with an `api_base` pointing at it, and every provider call the proxy makes is forwarded verbatim and written to a fixture bundle (default `tests/e2e/.fixtures`, override with `E2E_FIXTURE_DIR`). `E2E_FIXTURE_MODE=replay` runs the same tests against the same live proxy and database, but the edge answers the proxy's provider calls from the bundle instead of the provider, so the run makes zero provider calls and spends nothing while key auth, routing, cost calculation, and spend-log writes all still execute for real. Unset (or `live`) behaves exactly as before the knob existed. Both record and replay need the proxy up; only the provider is taken out of the loop

```bash
E2E_FIXTURE_MODE=record E2E_FIXTURE_DIR=/tmp/e2e-fixtures uv run pytest tests/e2e/quota_management/spend_tracking/test_provider_edge_spend_e2e.py -v
E2E_FIXTURE_MODE=replay E2E_FIXTURE_DIR=/tmp/e2e-fixtures uv run pytest tests/e2e/quota_management/spend_tracking/test_provider_edge_spend_e2e.py -v
```

Bundles stay local. `tests/e2e/.fixtures` is gitignored because a bundle holds verbatim provider response bodies and expires seven days after it was recorded, so record the suite you want before you replay it and never commit the result. CI keeps its bundle out of git too, as a private GitHub Actions artifact rather than a committed file, for the same reason

In CI the `.github/workflows/e2e_record_replay.yml` lane runs record and replay on a schedule. A Saturday cron records the `replayable` marker's tests against the real providers and publishes the bundle as a private `e2e-fixtures-bundle` artifact carrying a SHA-256 sidecar; weekday crons pull that artifact by its pinned digest, verify the checksum before extracting, and replay it with provider credentials deliberately set to bogus values, so a run that ever reached a real provider would fail instead of passing. An egress sentinel (`.github/scripts/e2e_egress_sentinel.py`) pins the provider hostnames to a local sink for the whole replay job and counts every connection that reaches them, and the job asserts that count is zero, so hermeticity is proven by measurement rather than by an absent bill. A red Saturday publishes no bundle, so the next weekday finds nothing fresh and fails loudly rather than replaying a week-old recording, and the seven-day freshness gate hard-fails any bundle that has drifted too far from the live providers. Run the lane on demand from the Actions tab with the `mode` input: `record` re-records and republishes, `replay` replays the current bundle. A test joins the lane by carrying `@pytest.mark.replayable` on top of its edge wiring, so add that marker only to a test whose provider traffic actually replays with zero egress

One sharp edge: a replayed response reuses the recorded provider response id, and that id is the primary key of `LiteLLM_SpendLogs`, so replaying against a database that still holds the record run's rows silently dedupes the spend writes and a spend assertion fails with zero rows. Run both commands above with `E2E_RESET_SPEND_LOGS=1` (and `DATABASE_URL` set in the pytest env) so each session truncates the spend log table after itself, or point replay at a fresh database

Another sharp edge, same root: record and replay derive every per-test token deterministically (the model name included, so a replay regenerates the exact requests the record run sent), which means an edge-wired deployment left in the database by an interrupted earlier run carries the same model name as the fresh one the current run registers. The proxy then holds two deployments under one model group and load-balances across both, and because the leftover's `api_base` points at the earlier run's edge process, which is gone, the calls that land on it fail with a connection error that reads like a transport bug rather than the stale row it is. Give each record or replay run a fresh database, or let a run finish so its own teardown deletes what it registered, and never reuse one long-lived proxy across back-to-back record/replay sessions. CI hands every job its own empty database and its own proxy, so it never sees this

Replay answers any provider call that drifted from the recording with an HTTP 599 whose body names the computed and closest recorded keys, so the test fails loudly instead of silently going live, and a bundle older than seven days fails at collection time naming its age; either way the fix is to re-record. Only tests that register edge-wired deployments participate: everything else hits its provider live in every mode, so record exactly the suite you replay. If the proxy runs in a container, set `E2E_PROVIDER_EDGE_ADVERTISE_HOST` (e.g. `host.docker.internal`) so the api_base the proxy stores can reach the edge on the pytest host, and `E2E_PROVIDER_EDGE_BIND_HOST=0.0.0.0` so the edge accepts it. The suites wired to the edge today are `quota_management/spend_tracking/test_provider_edge_spend_e2e.py`, `llm_translation/test_chat_completions_contract_e2e.py`, the OpenAI registrations in `llm_translation/test_embeddings_endpoint_e2e.py`, the Anthropic tests in `llm_translation/test_messages_e2e.py`, streamed and not, and the OpenAI batch deployment behind `batches/`. A streamed response replays as the chunk sequence the provider sent rather than one buffered body. See `CLAUDE.md` in this directory for the bundle format, the edge design, and the current limits (Bedrock). The scheduled CI record/replay lane is described above

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

1. Run `make lint-e2e-basedpyright` (or `make check` with your changes staged); the harness is fully typed and the gate allows zero basedpyright errors, enforced in CI on any PR touching `tests/e2e/**/*.py`

2. Add the models your test needs to the config your local proxy loads

3. Start the litellm proxy locally and run your suite against it:

   ```bash
   litellm --config <your-e2e-config>.yml --port 4000
   uv run pytest tests/e2e/<your_suite>/ -v
   ```

4. Capture screenshots of the test run and attach them to the PR as proof

5. If a test fails because it surfaced a real issue in the product, flag that explicitly in the PR rather than reworking the test until it passes
