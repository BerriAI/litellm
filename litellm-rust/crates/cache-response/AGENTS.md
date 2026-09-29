# Response caching

Design this crate for shared Rust execution used by the Python SDK and the Rust gateway. The Python SDK will remain, with more core execution moving to Rust and Python callbacks staying in Python. The Rust gateway is still evolving and is intended to replace the Python proxy. Keep response-cache policy independent of Python, HTTP serving, and either proxy's configuration format

Separate what is cached, how a hit is matched, and where entries are stored. Chat Completions, Messages, Responses, and embeddings are API workloads. Exact and semantic matching are lookup behaviors. Memory, Redis, disk, and object stores are storage choices. Embeddings are inference too, so do not use an inference-cache name to imply a category that excludes embeddings. Consult the existing Python cache and caching handler for behavior and compatibility contracts without copying their class structure

Storage traits, codecs, and backend capabilities belong in `litellm-cache` and the storage crates. Keep storage reusable for value types beyond LLM responses. This crate owns response entries, matching and freshness semantics, the Python-compatible response codec, and deferred-write policy. Core owns route-specific request identity, response encoding and reconstruction, embedding partial-hit orchestration, and stream capture and replay. Boundaries own configuration translation, resource construction, and caller identity

Construct and inject the response-cache service at the Python bridge or gateway boundary, as with the HTTP client. Reuse it across calls. Core and provider code must not discover cache configuration through Python globals, process configuration, or backend-specific factories

Keep `ResponseCache<B>` generic over its storage backend. Preserve typed backend contexts and capability bounds internally. Inject an object-safe service into core for runtime backend selection, so storage types do not spread through route and host types. Keep API request and response types statically typed. Add a generic parameter only where it preserves a useful type relationship or capability

Keep the core service contract narrow. Lookup and store must not require connection testing, ping, flush, deletion, counters, queues, or scripts. Require batch operations where a consumer needs partial hits, and keep management capabilities on their own interfaces. An exact-only adapter must remain explicit about its matching restriction. Supporting semantic matching requires a defined lookup-context and embedding execution contract, not just a renamed trait

Separate reusable resources from per-call policy. Backend configuration, namespace, default expiry, and entry limits belong to the configured service or backend. Read/write controls, expiry and freshness overrides, and authenticated caller scope belong to the call. Passing call options must not replace or mutate the route's configured service

Keep cache misses and storage failures distinguishable in return values. Core owns the decision to continue with provider execution after a cache failure. A read can reject an entry for freshness while the backend still retains it. Preserve the timestamp at which a response was produced when writing it later

Define lookup placement explicitly relative to authorization, deployment and credential resolution, and request-transforming callbacks. Cache identity must account for every input that affects reuse, including API surface and caller scope, while preserving intentional Python caching groups. Preserve existing keys and response formats unless changing them is an explicit migration decision

Cache normalized provider results before caller-specific response transformations. Hits must still run the applicable response processing, success callbacks, and cache-hit accounting. Keep callback execution in the host. Python cache implementations and semantic embedders that require the caller's task must use the existing host-operation mechanism rather than Python calls from a Rust worker. Preserve legacy fallback until that contract is supported

Keep unary caching independent of stream-only methods. Store streams only after successful exhaustion and protocol completion. Errors, incomplete streams, cancellation, and oversized entries must not populate the cache. Embedding batches need ordered partial results and reconstruction around the uncached inputs

Test each contract in its owner: storage capabilities in backend tests, envelopes and freshness here, reuse and replay in core, Python callback and fallback behavior at the bridge, and HTTP behavior at the gateway. Run backend contract checks and Python response-codec fixtures before exposing a new backend

`ScopedCache` requires an explicit shared or isolated scope at construction. `CacheOptions` has no default sharing policy. Callers may override policy per invocation without replacing the attached service. Versioned native envelopes reject incompatible API surfaces and versions as misses; this envelope is distinct from the legacy Python response codec

Response storage is not the source of budget or rate-limit coordination dependencies. Keep counters, reservations, and atomic admission operations out of `ResponseCacheService`, including when both services happen to use Redis
