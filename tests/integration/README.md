# Integration contracts

These tests exercise a running gateway, PostgreSQL and Redis with an owned local upstream. CircleCI owns this suite. Tests are grouped by behavior, with no automatic test retries or fallback to paid provider calls

The `cost` group runs the scripted-provider cost matrix through a dedicated sidecar. The sidecar serves the test-owned cost map over loopback through `LITELLM_MODEL_COST_MAP_URL`; cost goldens are checked into the integration suite and must not be copied into the E2E coverage registry

Use `tests/integration/run.py management`, `accounting`, `database`, `providers`, `extensions`, `sdk` or `cost` to run a selected group. Set `INTEGRATION_PROXY_URL`, `INTEGRATION_UPSTREAM_URL`, `INTEGRATION_MASTER_KEY` and `DATABASE_URL` to an isolated test deployment. The runner selects the new domain directories explicitly; the legacy OCI and sandbox selections remain separate

Management also requires `INTEGRATION_PEER_URL`, `REDIS_HOST` and `REDIS_PORT`. CircleCI starts two directly addressed proxy processes sharing only that job's stores. The test-only CLI wrapper supplies enterprise route entitlement, following the existing behavior suite's convention. It does not qualify license validation; run it with one worker and no reload

The generated lifecycle models use 20 examples, eight steps, generation and shrinking, with isolated resources per example. HTTP operation caps include generation and shrinking and exempt cleanup. Local qualification defaults to seed 4106601 and canonical order; CircleCI derives exploration and ordering seeds from the checked-out revision and workflow ID. Use `--seed` and `--order-seed` to reproduce a run. Actual installed Hypothesis version, settings, seeds and collected order are written beside the execution manifest

Reuse the existing canned provider handlers through `_support/upstream.py`. It rejects internal request fields and exposes actual received requests for independent assertions. Register every created resource for cleanup immediately, keep expected values independent of production calculations, and assert readback plus the runtime effect of a change

The CircleCI workflow starts its own database and Redis, restricts test-phase egress to its owned services and writes JUnit plus an executed-node manifest. Missing setup, skipped tests, failed cleanup or a selected test without a passed call fail qualification. Existing GitHub Actions jobs do not own these tests

Define integration contract IDs and their canonical test nodes in `contracts.json`. Every node must declare the same IDs with `covers`. The runner checks exact collected and passed selections against that mapping. These IDs belong to this CircleCI suite and must not be added to the separate E2E coverage registry. A manifest declaration alone does not mean a test passed

Provider sentinels currently use the controlled server, not live recordings. The provider shard also runs the existing strict replay controls for changed requests, exhausted interactions, leftover interactions and no provider connection. Future recorded scenarios must use that replay-only implementation; missing recordings cannot fall back to a real provider. The observation endpoint is destructive and the current selection runs serially against one owned upstream

Fixtures must contain synthetic data only. Keep private incident records and source documents out of code, fixtures, logs and PR descriptions

Database cases own their temporary schemas, roles, constraints and proxy processes. They prove reader-versus-writer execution with PostgreSQL lock observations, exercise real transaction wait limits and verify rollback after a reached database failure

Accounting cases compare persisted input and output cost components against literal rates, including zero and default prices. Cache state models assert actual upstream calls, response identity and every persisted charge. Generated accounting tests have a 180-second test limit to accommodate the asynchronous spend writer; CircleCI keeps other shards capped at 11 minutes and gives the cost shard 20 minutes

Provider contracts exercise actual TCP requests with synthetic credentials and local protocol peers. The S3 verifier uses independently implemented equations, a published known-answer vector, a fixed signing clock and deliberately invalid signed requests. Bedrock cases clear ambient AWS credential sources and check the literal model path, loaded role references, STS requests and bearer-only behavior

Streaming checks send real HTTP transfer chunks, including one-byte partitions, fragmented tools, incomplete transfers and a cancellation barrier. They assert meaningful text, tool arguments, final usage and persisted cost. The Redis recovery case owns a separate database and Redis process, uses the supported one-second circuit-breaker recovery setting, waits for the real subscriber and verifies response data in Redis after restart. CircleCI reuses its existing Redis image for that extra process; it never pulls an image during tests

The sdk shard exercises the SDK's own HTTP clients against local protocol peers with no gateway in the path, so a case here fails only when the client library or its wire behavior changes. The HTTP/2 case runs a hypercorn TLS peer offering h2 and http/1.1 over ALPN, drives the sync and async httpx handlers at it with `LITELLM_HTTP2` off and on, and asserts the version both the client and the peer observed on the wire. Put a test here only when it needs no proxy, database or Redis; a case that reaches the gateway belongs in one of the other shards

The extensions shard reuses the existing MCP arithmetic functions with a real SDK server, and uses the built-in generic callback and guardrail transports. It checks actual tool calls after saved edits, discovery preservation, malformed/error responses, callback correlation and credential exclusion, guardrail rewriting and denial, retained OpenAI consumers, persisted toolsets and A2A wire versions

Browser contracts live in `tests/e2e/ui/tests/integrationCritical` and run only through `tests/e2e/ui/integration.config.ts`. The CircleCI browser shard builds the checked-out dashboard, starts the owned proxy with that build, and verifies one exact browser result without retries or skips. The default Playwright selection excludes this directory. The focused project flow asserts the submitted create and clear values, fresh SQL state and actual blocked/restored serving while preserving model restrictions
