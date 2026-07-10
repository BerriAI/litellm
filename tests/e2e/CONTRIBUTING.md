# Contributors Guide

This directory holds the live end-to-end suites that prove product correctness against a real running proxy and real provider APIs. The goal of this guide is simple: when you ship a feature, you add e2e coverage that walks that feature the way production does, across every route and edge case it touches, so a later change that breaks it fails here first

Read this before adding a test and i recommend reading through CLAUDE.md

When contributing to this directory, please first discuss the change you wish to make via issue or pull request. We require screenshots and proof of your tests working on a live proxy. 


## Setup

The suites run against a live proxy, so bring one up first. `docker-compose.yml` here starts that proxy with a throwaway Postgres and Redis; `docker compose down -v` resets everything, so no state leaks between runs. The proxy config is inlined in the compose file under `configs`, prewired with example models (`gpt-5.5`, `claude-haiku-4-5`, `gemini-2.5-flash`, `openai-text-embedding-3-small`) whose keys come from your `.env`. If your test needs another model, a pricing override, or a guardrail declared up front, add it to that inline config and read it back in the test rather than hardcoding values

## Running the tests locally

1. Create a `.env` file in this directory with the provider keys the example models use:

   ```bash
   OPENAI_API_KEY="sk-..."
   ANTHROPIC_API_KEY="sk-..."
   GEMINI_API_KEY="..."
   ```

2. Bring the stack up from this directory:

   ```bash
   docker compose up -d
   curl -fs http://localhost:4000/health/liveliness
   ```

3. Run a suite against it; the harness reads `LITELLM_PROXY_URL` (default `http://localhost:4000`):

   ```bash
   uv run pytest tests/e2e/llm_translation/ -v
   ```

   The browser tests in the `management/` suite drive the dashboard the proxy serves at `/ui` through playwright, an optional dependency behind `importorskip` (the suite's API tests run without it). Install it once into your environment along with its browser:

   ```bash
   uv pip install playwright
   uv run playwright install chromium
   ```

   They also need a proxy whose bundled UI contains the change under test. The published `main-latest` image ships the UI from the last release; to test local UI changes, build the image from your branch and point the compose stack at it:

   ```bash
   docker build -t litellm-local .
   LITELLM_E2E_IMAGE=litellm-local docker compose up -d
   ```

4. Tear it down when you're done:

   ```bash
   docker compose down -v
   ```

Tests marked `@pytest.mark.e2e` skip when no proxy answers `/health/liveliness`, so a run that reports everything skipped means the stack isn't up, not that anything passed

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

## Pre-commit steps

Before you push

1. Run basedpyright over your changes; the harness is fully typed and new code must not add `Any` or widen the budgets

2. Add the models your test needs to the inline config in `docker-compose.yml`

3. Bring the stack up and run your suite against it:

   ```bash
   docker compose up -d
   uv run pytest tests/e2e/<your_suite>/ -v
   ```

4. Capture screenshots of the test run and attach them to the PR as proof

5. If a test fails because it surfaced a real issue in the product, flag that explicitly in the PR rather than reworking the test until it passes
