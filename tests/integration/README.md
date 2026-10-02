# Integration contracts

These tests exercise a running gateway, PostgreSQL and Redis with an owned local upstream. CircleCI owns this suite. Tests are grouped by behavior, with no automatic test retries or fallback to paid provider calls

The `cost` group is driven by `cost_tracking_cases.json`, which contains the cost map, literal requests, literal provider responses and expected accounting values. Each case has a name, contract ID, cost-map model, optional deployment overrides, request body, tagged response and exact or recount expectations. Request bodies use `$MODEL` for the registered proxy model, while responses use `$REQUEST_ID` for the per-run scenario ID. To add a case, add a cost-map entry when the model is new, add the request body and exact provider response data, and add hand-computed expected values. The upstream serves each stored response for any path under `/<scenario_id>`, while the test-owned cost map is served over loopback through `LITELLM_MODEL_COST_MAP_URL`

Use `tests/integration/run.py management`, `accounting`, `database`, `providers`, `extensions`, `mcp`, `sdk` or `cost` to run a selected group. The group to directory mapping is the `GROUPS` literal at the top of `run.py`; a new directory needs a `GROUPS` entry and an `OWNED_DIRECTORIES` entry in `_support/manifest.py`. Set `INTEGRATION_WORKERS` above 1 to run a group under pytest-xdist; the `mcp` job does this in CI, so MCP tests must own their resources per scenario. Set `INTEGRATION_PROXY_URL`, `INTEGRATION_UPSTREAM_URL`, `INTEGRATION_MASTER_KEY` and `DATABASE_URL` to an isolated test deployment. The runner selects the new domain directories explicitly; the legacy OCI and sandbox selections remain separate

Management also requires `INTEGRATION_PEER_URL`, `REDIS_HOST` and `REDIS_PORT`. CircleCI starts two directly addressed proxy processes sharing only that job's stores. The test-only CLI wrapper supplies enterprise route entitlement, following the existing behavior suite's convention. It does not qualify license validation; run it with one worker and no reload

The generated lifecycle models use 20 examples, eight steps, generation and shrinking, with isolated resources per example. HTTP operation caps include generation and shrinking and exempt cleanup. Local qualification defaults to seed 4106601 and canonical order; CircleCI derives exploration and ordering seeds from the checked-out revision and workflow ID. The ordering seed shuffles the file order and the test order inside each file but keeps each file's tests together, so module fixtures are built once per file. Use `--seed` and `--order-seed` to reproduce a run. Actual installed Hypothesis version, settings, seeds and collected order are written beside the execution manifest

Reuse the existing canned provider handlers through `_support/upstream.py`. It rejects internal request fields and exposes actual received requests for independent assertions. Register every created resource for cleanup immediately, keep expected values independent of production calculations, and assert readback plus the runtime effect of a change

The CircleCI workflow starts its own database and Redis, restricts test-phase egress to its owned services and writes JUnit plus an executed-node manifest. Missing setup, failed cleanup or a selected test with neither a passed call nor a skip fail qualification. Skipped nodes are listed under `skipped` in `execution.json`, so the skip reasons double as the open bug list. Existing GitHub Actions jobs do not own these tests

Groups other than MCP have no per-node manifest. Those groups fail when pytest fails, when collection errors, or when a selected file collects zero tests. MCP additionally enforces the required baseline described below. Older tests still carry `@pytest.mark.covers(...)` decorators; the marker stays registered so they collect, but the IDs are not checked against anything and new tests should not use it. The GitHub Actions coverage census reads the `GROUPS` literal in `run.py` and treats every `tests/integration/<directory>/test_*.py` file in a scheduled group as owned by CircleCI

Provider sentinels currently use the controlled server, not live recordings. The provider shard also runs the existing strict replay controls for changed requests, exhausted interactions, leftover interactions and no provider connection. Future recorded scenarios must use that replay-only implementation; missing recordings cannot fall back to a real provider. The observation endpoint is destructive and the current selection runs serially against one owned upstream

Fixtures must contain synthetic data only. Keep private incident records and source documents out of code, fixtures, logs and PR descriptions

Database cases own their temporary schemas, roles, constraints and proxy processes. They prove reader-versus-writer execution with PostgreSQL lock observations, exercise real transaction wait limits and verify rollback after a reached database failure

Accounting cases compare persisted input and output cost components against literal rates, including zero and default prices. Cache state models assert actual upstream calls, response identity and every persisted charge. Generated accounting tests have a 180-second test limit to accommodate the asynchronous spend writer

Provider contracts exercise actual TCP requests with synthetic credentials and local protocol peers. The S3 verifier uses independently implemented equations, a published known-answer vector, a fixed signing clock and deliberately invalid signed requests. Bedrock cases clear ambient AWS credential sources and check the literal model path, loaded role references, STS requests and bearer-only behavior

Streaming checks send real HTTP transfer chunks, including one-byte partitions, fragmented tools, incomplete transfers and a cancellation barrier. They assert meaningful text, tool arguments, final usage and persisted cost. The Redis recovery case owns a separate database and Redis process, uses the supported one-second circuit-breaker recovery setting, waits for the real subscriber and verifies response data in Redis after restart. CircleCI reuses its existing Redis image for that extra process; it never pulls an image during tests

The `messages_endpoint/` directory holds `/v1/messages` endpoint contracts: native-provider backends under `providers/` (`anthropic`, `bedrock`, `gemini`) and the translation bridges (`responses_bridge`, `chat_bridge`) at the top level. It runs in the providers shard; `run.py` selects test files recursively under each scheduled directory

The sdk shard exercises the SDK's own HTTP clients against local protocol peers with no gateway in the path, so a case here fails only when the client library or its wire behavior changes. The HTTP/2 case runs a hypercorn TLS peer offering h2 and http/1.1 over ALPN, drives the sync and async httpx handlers at it with `LITELLM_HTTP2` off and on, and asserts the version both the client and the peer observed on the wire. Put a test here only when it needs no proxy or database. CircleCI starts a local Redis for this shard like the others, so SDK-side caching cases that need a real Redis server belong here too; a case that reaches the gateway belongs in one of the other shards

The extensions shard uses the built-in generic callback and guardrail transports. It checks callback correlation and credential exclusion, guardrail rewriting and denial, retained OpenAI consumers and A2A wire versions. CircleCI runs it on parallel nodes, and each node starts its own database, Redis, upstream and proxy and runs its share of the group's files serially, split by recorded timings with `circleci tests split`. Tests keep the isolation of a serial run; they still must not assume a particular set of sibling files. `run.py <group> --list` prints a group's files and `run.py <group> <file>...` runs a subset of them

The mcp shard runs the MCP gateway against SDK peers owned by each test (`_support/mcp.py`): streamable HTTP, SSE and stdio peers, an OpenAPI-spec app, and an OAuth 2.1 authorization-server double. Every peer records the requests it receives so a test can assert what reached the peer, not only what the proxy answered. The shard runs with `INTEGRATION_WORKERS` set and with `INTEGRATION_COVERAGE=1`, which starts the proxy under `coverage run --parallel-mode` limited to the MCP modules and stores `coverage.txt` plus an HTML report with the job artifacts. A test that fails because the product is wrong is skipped with `pytest.skip("BUG: <symptom>")` so the skip list in `execution.json` is the open MCP bug list

Browser contracts live in `tests/e2e/ui/tests/integrationCritical` and run only through `tests/e2e/ui/integration.config.ts`. The expected browser results are listed in `expected.json` in that directory and checked by `.circleci/scripts/verify_integration_browser.py`. The CircleCI browser shard builds the checked-out dashboard, starts the owned proxy with that build, and verifies one exact browser result without retries or skips. The default Playwright selection excludes this directory. The focused project flow asserts the submitted create and clear values, fresh SQL state and actual blocked/restored serving while preserving model restrictions

Two always-on `-replica` CircleCI jobs (management, database) run their groups in replica mode, where every proxy connects through a real `litellm_writer` role and a real read-only `litellm_reader` role against the same PostgreSQL. Nothing is captured there: the job passes when the tests pass, and a write routed to the read-only reader fails the test that issued it. A deeper check runs on demand as the `routing_parity` workflow, triggered through the CircleCI API v2 pipeline endpoint on the PR branch with `{"parameters": {"routing_parity_base": "<40-hex merge-base sha>"}}`. The workflow fans out over the seven groups, and each `routing-parity-<group>` job runs its own group twice against the same test harness, once with `litellm/`, `enterprise/`, and `litellm-proxy-extras/` checked out from the base revision and once from the head, with a pytest plugin snapshotting `pg_stat_statements` into `routing-observed.json` per side. The `check` step then compares the two observations and writes `routing-diff.txt`: a statement seen on both sides fails when its role set changed, globally or for the same test (per-test capture is skipped under xdist), unless it is listed in `tests/integration/routing/either_role.json`, where each entry names the statement and a one-line reason it legitimately runs on whichever role asks for it, printed under `== either role ==`. Queries seen on only one side are listed, never failed, `pg_stat_statements` evictions and a role that never ran a statement are failures


### MCP regression and conformance baseline

This gate protects existing behavior while the stateless refactor proceeds. Completion establishes a
passing, enforced baseline; it does not certify exhaustive conformance or future capabilities.
Each later implementation preserves this baseline and adds tests for the behavior it changes

The `mcp` group runs a pinned official conformance client against the official reference
both directly and through a source-built gateway. Install it with
`MCP_CONFORMANCE_ROOT=/tmp/mcp-conformance bash .circleci/scripts/install_mcp_conformance.sh`
after the existing integration dependency setup, then use the normal MCP integration command.
The destination must be fresh. The installer verifies each immutable archive before extraction
and uses both upstream lockfiles. CircleCI installs this automatically and publishes runner logs,
raw checks, negotiation observations and helper coverage with the existing integration artifacts.

The runner is pinned to `7169291ec0b68eb370fddcd9947313ab0d5e4156`; the unmodified legacy
reference is pinned separately to `8f3994c75ff1aed1e39f91cff9358e2bc2c81dcd`. The newer
reference mistakes a valid legacy initialize containing `_meta: {}` for stateless traffic.
Full download URLs and archive SHA-256 values live in the installer.

Coverage is deliberately explicit:

- The official client negotiates 2025-11-25. Its `--spec-version` flag selects assertions;
  it does not pin its SDK handshake. Official cases cover each declared upstream revision.
- Session termination is a separate mandatory gateway contract. Its raw HTTP control uses
  the current pinned reference, which correctly returns 404 after deletion. The legacy
  reference returns 400 for that case. Both references run unmodified with their own lockfiles.
- The SDK matrix supplies the older-client gap: all 16 declared ordered revision pairs,
  HTTP and SSE ingress, and HTTP, SSE and stdio upstreams. It checks the seven operations in
  `capabilities.py` and records requested and returned revisions on both connections.
- The official simple-text scenario accepts error text, and its reference rejects its omitted
  arguments. Exact text with `{}` and omitted-argument forwarding therefore have explicit SDK
  cases. This limitation cannot be treated as a successful official simple-text result.
- The schema scenario looks up an unprefixed fixture name. A direct/gateway SDK comparison
  instead requires the full JSON Schema 2020-12 input schema to survive unchanged.
- The official DNS-rebinding scenario explicitly targets unauthenticated localhost servers.
  This authenticated gateway instead has explicit allowed/denied Origin execution cases with
  `LITELLM_CORS_ORIGINS` configured, covering HTTP and SSE. The default wildcard is unchanged.
- Logging, completion, resource subscriptions, sampling and elicitation are not advertised by
  this gateway contract; their capability-specific scenarios do not establish legacy support.
  Modern 2026-07-28 and extension scenarios belong to later activation gates.

The adapter changes only authentication and the fixture name to the gateway's advertised prefix.
It preserves protocol versions, arguments, metadata, Origin/Host headers and response bytes.
No discovery warm-up or expected-failure exemption is used. Missing, failed, skipped or wrongly
negotiated required cases fail the MCP integration result.

`mcp/conformance_baseline.json` is the independent required-coverage contract. Its 19 official
scenarios across four upstream revisions, 96 SDK revision/transport combinations, and nine explicit
checks require 181 passing nodes. Test generation does not read this file, so removing a scenario,
transport case or test file cannot silently remove its requirement. Capability mappings name the
required test families that exercise each revision's operations and extensions. A completed
revision or capability without mapped required coverage fails the gate

To extend the gate, add meaningful behavior assertions to the existing mapped integration file,
then update the baseline with the required cases and corresponding capability mapping in the same
PR. Review that the mapped test actually exercises the capability. A mapping alone is not proof.
Keep required cases passing without skip/xfail exemptions. Track an existing uncovered requirement
with its owning ticket; do not remove baseline protection or advertise unverified new support

CircleCI currently runs MCP on one node with four pytest workers. The pytest controller combines
all worker reports into `execution.json`; the full invocation checks all required nodes. Missing
worker results, absent files, skips and incomplete runs fail the baseline. Local file selections
check the required cases in those files and are not evidence of a complete gate. If MCP is later
split across CircleCI nodes, aggregate their results against the entire baseline before accepting
the job; per-file successes alone are insufficient

Remaining coverage grows with the owning implementation:

| Behavior | Owner and acceptance boundary |
| --- | --- |
| Modern upstream calls without initialization; affected legacy auth/policy parity | LIT-7745, tested with its implementation |
| Authorized list/call across cold replicas without session affinity | LIT-4500, tested with its implementation |
| Full schema/result preservation and remaining upstream pagination | LIT-7750 and LIT-5594 |
| Input relay, MRTR and event/cancellation behavior | LIT-4508, LIT-7752 and LIT-4510 |
| Expanded OAuth and security regressions | LIT-3467, LIT-3559 and LIT-4506 alongside the affected fixes |
| Applicable combined conformance, canary and rollback | LIT-8305 before the corresponding release |

The SDK matrix provides successful legacy-operation compatibility coverage. It does not repeat
every official rich-content, error, progress or lifecycle assertion for every older client and
transport. Expand those cases when relevant paths change; this limitation does not block unrelated
implementation. Modern revision and extension activation still requires their applicable tests

An installed workflow alone is not an enforced merge gate. Require the hosted MCP job's exact
status in the existing main ruleset. Close LIT-7744 after this baseline is merged, passes and is
required. LIT-7745 can be developed alongside gate completion; its shared-path changes must pass
the baseline and their added legacy/modern tests before merging. Later coverage extensions belong
to their implementation tickets rather than keeping LIT-7744 open indefinitely

The `ci/circleci: integration-mcp` status is published for every normal PR pipeline. Before installing
dependencies or starting services, its `mcp-integration` path filter completes the job successfully
as "not applicable" for changes confined to documentation, frontend files unrelated to MCP, or
unrelated tests. MCP files, shared test fixtures and integration helpers always run the full gate.
Runtime code, dependencies, CI configuration and unrecognized paths also run it conservatively.
Non-PR pipelines, unavailable merge bases, empty diffs and failed or invalid classification run the
suite. Renames include both old and new paths. Filtering selects whether to run the entire job;
it never reduces the 181 required cases for an applicable run
