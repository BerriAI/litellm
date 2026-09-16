# Shared provider-response cache

`E2E_PROVIDER_CACHE=1` enables automatic response reuse in the live E2E mode. Standard OpenAI and Anthropic model registrations use the provider edge, as do Anthropic-on-Bedrock registrations that carry no AWS identity of their own. Existing custom API bases, named credentials, mocked models and realtime WebSocket deployments keep their existing routing. Other provider protocols remain live

The edge caches complete successful POST responses for `/v1/chat/completions`, `/v1/messages`, `/v1/embeddings` and `/v1/responses` on the OpenAI and Anthropic mounts, SSE streams included, and for `/model/{id}/converse` and `/model/{id}/invoke` on a Bedrock mount. Unsupported endpoints pass through. Each endpoint family has its own completeness rule, so a truncated embedding or a Responses run that never reached `response.completed` is not stored

Bedrock's streaming endpoints, `converse-stream` and `invoke-with-response-stream`, are not cacheable. They still cross the edge and are still re-signed, so they need the same IAM, but they always call the provider. AWS frames them as binary `vnd.amazon.eventstream` rather than SSE, and reading a terminal event out of that is what a completeness rule for them would need. That matters more than the endpoint count suggests: the Claude Code compat cells drive the real CLI, which always streams, so most Bedrock traffic in the suite is not cached today

## Request identity

A recording belongs to one test. The key is a keyed digest over the test's node id, the method, the URL, the effective outbound headers (including authentication and HTTP-library defaults), body presence and the body bytes, with one normalization: a 12-hex-digit run, the shape `unique_marker()` mints, is replaced by a placeholder in both the URL and a UTF-8 body. Nothing else is normalized away. No prompts, JSON values or credentials are rewritten, and the rule is the one `fixture_canonical.py` already applies for record/replay, so there is a single definition of what a marker is

Requests that differ only by their markers therefore share a canonical identity, which is what makes the cache reusable across builds: every e2e test salts its prompt afresh, so an exact-byte key would miss on every call. Within one test, calls that share a canonical identity are still recorded and replayed separately, by a FIFO slot index appended to the key. That matters because a replayed response carries the recorded provider response id, `LiteLLM_SpendLogs.request_id` is that id, and one shared recording answering two calls would collapse two spend rows into one

Two different tests never share a recording, and a provider call made outside any test (fixtures, session setup) is never cached, because the identity has no test node id to bind to

Provider `Set-Cookie` headers are dropped before validation and never recorded: the edge already withholds them from the proxy, and OpenAI responses always carry Cloudflare bot-management cookies

An eligible miss calls the provider. A complete successful response is stored immediately even if a later test assertion fails. Provider errors, malformed responses, truncated streams and cancelled captures are not stored. Cache reads, writes and lease failures fall through to normal provider behavior; they introduce no provider retry. An already-started response cannot be restarted after a delivery failure

## Bedrock

Bedrock could not be mounted before because SigV4 signs the `Host` header, so a rewritten `api_base` failed signature verification at the provider. The edge now re-signs: it drops the proxy's signature headers, signs the upstream request with the run pod's own AWS identity from its EKS Pod Identity association, and forwards that. The signature headers are excluded from the key, since `x-amz-date` is a timestamp and keying on it would make every Bedrock call a permanent miss

Almost every Bedrock deployment in the suite declares its region as `os.environ/AWS_REGION`, which only the proxy can resolve, and the run pod does not share that environment. A `us.` inference profile fans out across the US regions and is reachable from any of them, so those route to the default mount whatever the proxy resolved. A model that is not cross-region and declares its region that way keeps its direct path rather than being sent to a region it may not exist in.

Only deployments that carry no AWS identity of their own route to the edge. A deployment with `aws_role_name`, `aws_access_key_id`, an `api_base` or an `aws_bedrock_runtime_endpoint` keeps its direct path, because re-signing it would quietly replace the very credential chain that test exists to prove

Which models route is an explicit allowlist in `provider_cache_routing.py`, mirroring the runner role's IAM policy, which names its models one by one. That coupling is deliberate: the edge re-signs with the run pod's identity, so a model the role cannot invoke comes back 403 from Bedrock rather than falling back. An unlisted model keeps its direct path and loses only caching, so adding a Bedrock model to the suite can never turn it red. Adding one to the edge is a policy edit in litellm-ops plus a line here

Vertex and Gemini are not mounted. litellm's `_check_custom_proxy` rewrites a path-prefixed Vertex `api_base` into `{api_base}:{endpoint}`, dropping project, location and model, so a mount under a path prefix cannot work without either a root-mounted edge on its own port or a change in litellm

Recordings are shared across workers and builds through dedicated Redis, separate from the candidate's own cache. They expire 86,400 seconds after capture starts, based on Redis time. Reads never extend expiry. There is no scheduled recapture: the next miss calls the provider again. Bounded coordination reduces duplicate concurrent calls, but slow or failed captures may lead to extra live calls after the wait expires

## Configuration

The trusted runner receives:

- `E2E_PROVIDER_CACHE`: `1` to enable, `0` to use the normal live path
- `E2E_PROVIDER_CACHE_REDIS_URL`: authenticated dedicated Redis URL
- `E2E_PROVIDER_CACHE_HMAC_KEY`: dedicated secret containing at least 32 bytes
- `E2E_PROVIDER_CACHE_NAMESPACE`: shared environment namespace, independent of build and candidate revision
- `E2E_PROVIDER_CACHE_METRICS_DIR`: optional per-process counter artifact directory

Do not give cache credentials to candidate deployments. Counter artifacts contain no recorded payloads or credentials. Hits count shared-cache responses; upstream attempts count actual forwards from the edge. Every counter is emitted twice, once as a flat total and once under `mount:{mount}:`, so a hit rate can be read per provider rather than only in aggregate. Existing application-cache observations still count requests arriving at the edge, including shared-cache hits

Tests that require real provider timing, limits or state use `@pytest.mark.provider_live`. The marker keeps newly registered models on live routes without weakening their assertions. The provider prompt-caching tests carry it because a replayed priming response reports cache creation rather than a cache read. Ordinary assertion failures still fail E2E. The shared cache does not modify provider response IDs or make the proxy aware of replay

## Recorded response semantics

Replay preserves the original response ID, usage and end-to-end headers. The proxy can therefore deduplicate repeated provider IDs when storing spend-log rows, just as it does when a live upstream returns the same ID twice. One spend-log row per invocation is not guaranteed for identical recorded responses. Spend reconciliation keeps its distinct-ID and row-count assertions: its prompts differ by an index as well as a marker, so they stay distinct once markers are normalized, and calls that are canonically equal within one test take separate FIFO slots and separate recordings anyway. Accounting tests are not automatically excluded from caching

Provider remaining-quota headers describe the captured response. Metrics derived from them are historical on a cache hit, not a measurement of current provider capacity. Gateway-generated API-key quota headers are a separate contract. A test of fresh provider quota or timing must use the live-provider policy; replay can still exercise how the proxy processes the recorded headers

## Qualification

`tests/code_coverage_tests/test_provider_cache.py` exercises local HTTP providers and disposable real Redis, including the marker-canonical key, the FIFO slot index, per-test isolation, SigV4 re-signing against a local upstream, and each endpoint's completeness rule. CI runs these checks with the existing provider-edge and replay harness tests. These component checks do not establish Buildkite deployment, full-suite cross-build reuse or a genuine 24-hour expiry observation; those require separate runtime evidence
