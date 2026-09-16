# Shared provider-response cache

`E2E_PROVIDER_CACHE=1` enables automatic response reuse in the live E2E mode. Standard OpenAI and Anthropic model registrations use the provider edge. Existing custom API bases, named credentials, mocked models and realtime WebSocket deployments keep their existing routing. Other provider protocols remain live

The edge caches complete successful POST responses for `/v1/chat/completions` and `/v1/messages`, including streams. Unsupported endpoints pass through. It matches the method, original URL, effective outbound headers (including authentication and HTTP-library defaults), body presence and exact body bytes using a full keyed digest. It sends the same prepared request used for matching. No prompts, random markers, JSON values or credentials are normalized away

An eligible miss calls the provider. A complete successful response is stored immediately even if a later test assertion fails. Provider errors, malformed responses, truncated streams and cancelled captures are not stored. Cache reads, writes and lease failures fall through to normal provider behavior; they introduce no provider retry. An already-started response cannot be restarted after a delivery failure

Recordings are shared across workers and builds through dedicated Redis, separate from the candidate's own cache. They expire 86,400 seconds after capture starts, based on Redis time. Reads never extend expiry. There is no scheduled recapture: the next miss calls the provider again. Bounded coordination reduces duplicate concurrent calls, but slow or failed captures may lead to extra live calls after the wait expires

## Configuration

The trusted runner receives:

- `E2E_PROVIDER_CACHE`: `1` to enable, `0` to use the normal live path
- `E2E_PROVIDER_CACHE_REDIS_URL`: authenticated dedicated Redis URL
- `E2E_PROVIDER_CACHE_HMAC_KEY`: dedicated secret containing at least 32 bytes
- `E2E_PROVIDER_CACHE_NAMESPACE`: shared environment namespace, independent of build and candidate revision
- `E2E_PROVIDER_CACHE_METRICS_DIR`: optional per-process counter artifact directory

Do not give cache credentials to candidate deployments. Counter artifacts contain no recorded payloads or credentials. Hits count shared-cache responses; upstream attempts count actual forwards from the edge. Existing application-cache observations still count requests arriving at the edge, including shared-cache hits

Tests that require real provider timing, limits or state use `@pytest.mark.provider_live`. The marker keeps newly registered models on live routes without weakening their assertions. Ordinary assertion failures still fail E2E. The shared cache does not modify provider response IDs or make the proxy aware of replay

## Recorded response semantics

Replay preserves the original response ID, usage and end-to-end headers. The proxy can therefore deduplicate repeated provider IDs when storing spend-log rows, just as it does when a live upstream returns the same ID twice. One spend-log row per invocation is not guaranteed for identical recorded responses. Existing spend reconciliation requests use distinct prompt markers and retain their distinct-ID and row-count assertions; accounting tests are not automatically excluded from caching

Provider remaining-quota headers describe the captured response. Metrics derived from them are historical on a cache hit, not a measurement of current provider capacity. Gateway-generated API-key quota headers are a separate contract. A test of fresh provider quota or timing must use the live-provider policy; replay can still exercise how the proxy processes the recorded headers

## Qualification

`tests/code_coverage_tests/test_provider_cache.py` exercises local HTTP providers and disposable real Redis. CI runs these checks with the existing provider-edge and replay harness tests. These component checks do not establish Buildkite deployment, full-suite cross-build reuse or a genuine 24-hour expiry observation; those require separate runtime evidence
