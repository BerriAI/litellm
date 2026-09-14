# Integration contracts

These tests exercise a running gateway, PostgreSQL and Redis with an owned local upstream. CircleCI owns this suite. Tests are grouped by behavior, with no automatic test retries or fallback to paid provider calls

Use `tests/integration/run.py management`, `accounting` or `providers` to run a selected group. Set `INTEGRATION_PROXY_URL`, `INTEGRATION_UPSTREAM_URL`, `INTEGRATION_MASTER_KEY` and `DATABASE_URL` to an isolated test deployment. The runner selects the new domain directories explicitly; the legacy OCI and sandbox selections remain separate

Reuse the existing canned provider handlers through `_support/upstream.py`. It rejects internal request fields and exposes actual received requests for independent assertions. Register every created resource for cleanup immediately, keep expected values independent of production calculations, and assert readback plus the runtime effect of a change

The CircleCI workflow starts its own database and Redis, restricts test-phase egress to its owned services and writes JUnit plus an executed-node manifest. Missing setup, skipped tests, failed cleanup or a selected test without a passed call fail qualification. Existing GitHub Actions jobs do not own these tests

Add contract definitions to the existing `tests/e2e/coverage_registry` and map canonical node IDs to those definitions in `contracts.json`. Every node must declare the same IDs with `covers`. The runner checks exact collected and passed selections against that mapping. Registry declarations alone do not mean a test passed

Provider sentinels currently use the controlled server, not live recordings. The provider shard also runs the existing strict replay controls for changed requests, exhausted interactions, leftover interactions and no provider connection. Future recorded scenarios must use that replay-only implementation; missing recordings cannot fall back to a real provider. The observation endpoint is destructive and the current selection runs serially against one owned upstream

Fixtures must contain synthetic data only. Keep private incident records and source documents out of code, fixtures, logs and PR descriptions
