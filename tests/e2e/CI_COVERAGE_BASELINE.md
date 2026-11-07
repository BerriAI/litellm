# CI vs e2e coverage baseline

Measured 2026-07-06 on `litellm_internal_staging`. Purpose: a standing comparison
between the legacy CI test volume and the live e2e suite, so decisions about
where new tests go (and which CI suites are not earning their runtime) are made
against numbers instead of memory. Re-measure by re-running the counts below
after major suite changes and update this file in the same PR

Counting method: e2e counts are collected tests (`uv run pytest tests/e2e/<dir>
--collect-only -q`, includes parametrization). CI counts are `def test_`
functions per directory (`grep -rcE "^\s*(async )?def test_"`), attributed to
the feature whose code they exercise. UI counts are `it(`/`test(` blocks in
`ui/litellm-dashboard`

## Headline

Roughly 29,400 backend CI test functions plus 4,100 UI test blocks, against 163
collected e2e tests in 10 active suites (about 0.5%). 7 of 18 features had zero
dedicated e2e coverage at measurement time

## Why CI volume was not catching production breakage

The 2026-07-06 stage run failed 46 tests and errored 17 more while every legacy
CI suite was green. The causes were, in order of blast radius: a harness routing
bug sending `/model/new` to the data plane (404, took down all runtime model
registration), a missing `Gateway.create_model` method (AttributeError at
fixture setup, all batch tests), and an exhausted provider account. None of
these are visible to mocked unit suites: they live in deployment topology,
integration wiring, and account state. That is the gap this e2e suite exists to
close, and the class of test worth adding here rather than to CI

## Per-feature counts (e2e ascending, at measurement time)

| Feature | e2e tests | CI unit tests | CI directories attributed |
|---|---:|---:|---|
| Guardrails and policy engine | 0 | ~2,337 | `test_litellm/proxy/guardrails`, `guardrails_tests`, `proxy/policy_engine` |
| Key/team/user/org management | 0 (now 8, PR #32300) | ~1,727 | `proxy/management_endpoints`, `management_helpers`, `proxy_admin_ui_tests`, root key/team/user/org tests |
| MCP gateway | 0 | ~1,654 | `proxy/_experimental/mcp_server`, `mcp_tests`, `experimental_mcp_client` |
| Caching | 0 | ~363 | `test_litellm/caching`, `local_testing` cache tests |
| Vector stores / RAG | 0 | ~206 | `proxy/vector_store_endpoints`, `search_tests`, `vector_store_tests`, `rag_endpoints` |
| Model management | 0 (now 4, PR #32272) | ~101 | `test_model_management_endpoints.py`, `store_model_in_db_tests`, root `test_models.py` |
| UI dashboard | 0 in harness (56 playwright separate) | ~4,096 | `ui/litellm-dashboard` src/tests + `e2e_tests` |
| Responses API and /v1/messages | 2 | ~481 | `test_litellm/responses`, `llm_responses_api_testing`, anthropic endpoint dirs |
| Logging integrations | 3 | ~2,168 | `test_litellm/integrations`, `logging_callback_tests`, `otel_tests` |
| Auth / access control | 3 | ~783 | `proxy/auth`, `proxy_security_tests` |
| Media (images/audio/OCR/video) | 3 | ~261 | `image_gen_tests`, `audio_tests`, `ocr_tests`, `test_litellm/ocr`, video dirs |
| Embeddings and rerank | 7 | ~42 | `local_testing` embedding tests |
| Pass-through endpoints | 7 | ~590 | `proxy/pass_through_endpoints`, `pass_through_unit_tests`, `pass_through_tests` |
| Router (fallbacks/limits/strategies) | 9 | ~1,276 | `router_strategy`, `router_unit_tests`, root `test_router*`, `router_utils` |
| Realtime | 10 | ~37 | `test_litellm/realtime_api`, `proxy/realtime_endpoints` |
| LLM translation / provider adapters | ~12 | ~9,373 | `test_litellm/llms`, `tests/llm_translation`, `litellm_core_utils` |
| Batches and files | 17 | ~254 | `test_litellm/batches`, `proxy/batches_endpoints`, `openai_files_endpoint`, `batches_tests` |
| Spend tracking and budgets | 71 | ~354 | `proxy/spend_tracking`, `spend_tracking_tests`, budget/rate-limiter hooks |

## What the CI trees cannot express (do not port these)

Most of the legacy volume asserts implementation internals that have no
observable HTTP contract: SQL string construction, payload sanitizer edge
cases, mocked provider adapters, settings CRUD against mocked DB clients. Six
of seven files in `test_litellm/proxy/spend_tracking` use mocks. Porting these
to e2e adds cost and flake without new signal; the e2e suite's job is the
contracts mocks structurally cannot verify: deployment topology, cross-plane
routing, DB and cache behavior under real writes, provider account state,
enforcement timing (auth-cache TTL, data-plane sync), and spend arithmetic on
real responses

## Standing priorities derived from this baseline

Zero-e2e features that need stage configuration before they are testable:
guardrails (live guardrail provider), MCP gateway (MCP servers). Testable
without config changes: caching. Cross-cutting suites planned from this
baseline: config/UI parity (drift between stage and what production and the
dashboard expect), then golden user journeys (multi-feature production
narratives). A deliberately rejected idea, for the record: a scripted fake
upstream for provider failure simulation was considered and declined 2026-07-06
