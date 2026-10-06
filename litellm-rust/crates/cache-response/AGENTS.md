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

`ScopedCache` requires an explicit shared or isolated scope at construction. Per-call `CachePolicy` controls reads, writes, expiry, and freshness without replacing the attached scope or service. `CacheOptions` binds that policy to an explicit scope for storage requests and has no default sharing policy. Versioned native envelopes reject incompatible API surfaces and versions as misses; this envelope is distinct from the legacy Python response codec

Response storage is not the source of budget or rate-limit coordination dependencies. Keep counters, reservations, and atomic admission operations out of `ResponseCacheService`, including when both services happen to use Redis

## Boundary

- Owns `ResponseCache<B>`, key derivation, scope and per-call policy, envelopes, entry freshness, the Python-compatible response codec, deferred writes, and the object-safe views over `ResponseCache<B>` (`ResponseCacheService`, the exact-only `ExactResponseCache`, `ConnectionProbe`)
- Having only one consumer today, such as the Python bridge, is not a reason to move code out. This crate serves both the SDK and the gateway
- Python wire shapes stay out: decoding the legacy key flags (`api_parameter`, `internal_parameter`) and the legacy activation flags (`supported_call_type`, `configured`, `default_on`, `use_cache`) is configuration translation owned by `python-bridge/src/cache/native`
- A request carries one resolved read/write policy. The gateway and the Python bridge each translate their inputs into it; don't add parallel control types alongside `CachePolicy`
- The native keying layout stays in `CacheKey::derive` in `key.rs`; changing it is a key migration, so bump `KEY_VERSION` with it

## Keys

Key construction is coupled to storage. The storage that holds an entry is the only thing that derives its key. Python-backed storage gets its key from the Python cache's `get_cache_key(**kwargs)`. Native storage gets its key from `litellm-cache-response`. Never derive a key in Rust for Python-backed storage, and never ask Python for a native backend's key

The two paths do not need byte-identical keys. A native key may hash typed JSON, the API surface, and the caller scope under a versioned namespace, while Python hashes formatted kwargs. Both paths must use the same precedence for presets, model and caching groups, files, namespaces, and semantic tenant scope. When the two paths disagree, the native key must be the stricter one: a divergence may cost hit rate but must never return a response cached for a different request. Native keys hash every parameter the host forwards, so they always include provider-specific parameters, as if Python's `enable_caching_on_provider_specific_optional_params` were on. litellm-owned kwargs are removed by the host before they reach key material, not filtered here

A native key identifies the model the way Python's `_get_model_param_value` does: `CacheTarget::ModelGroup` when the host routed through a model group, `CacheTarget::Model` otherwise. The host resolves the group, applying caching groups first where it supports them, and passes it as `litellm_core::CallOptions::model_group`. Deployment URLs and credentials never enter the key, so every deployment of one group shares entries and two groups never do

A request that `before_provider_request` changed skips the cache on both storage paths. Core decides this in `CacheRequest::from_logical` before any key is derived, so neither derivation sees the rewritten request

### Structure

Make the wrong path impossible to write rather than documented

- `ResponseCacheService::key` is a required method with no default implementation. Each storage adapter chooses its derivation explicitly
- `ResponseCacheRequest` carries key material (`CacheKeyInput`, surface, scope) and never a key. Nothing upstream of the service fills in a key or a precomputed key input
- Native derivation (`CacheKey::derive`) is private to `litellm-cache-response`. Adapters outside the crate cannot call it
- Key material carries no per-field participation. If a native path ever needs to drop parameters from the key, the rule lives in this crate and is injected when `ResponseCache<B>` is constructed, never decided per request
- The resolved key travels as a `CacheKey` value passed to `lookup` and `store`. `CacheKeyInput::Preset` means only the caller's `preset_cache_key`; do not write a resolved key back into the request
- Python-backed storage accepts only `CacheScope::Shared`, because Python's key cannot carry a caller scope. Reject `Isolated` at construction instead of silently dropping it
- `litellm-cache-response` has no pyo3 dependency. The Python adapter in `python-bridge/src/cache/python` asks Python for the key and does not project kwargs into Rust key input

Do not reintroduce side channels such as a precomputed key input or key context on `CacheOptions` or `ScopedCache`. If a route needs a new input to affect the key, add it to the key material and teach both derivations about it

### Tests

Types catch the accidental mistakes. Tests catch drift and deliberate workarounds

- Delegation: a Python cache subclass whose `get_cache_key` returns a fixed key must be the key used for lookup, store, and the cache-key response header on native inference
- Strictness: for each parameter class and both provider opt-in values, a parameter that changes the Python key also changes the native key
- Native `CacheKey::derive` precedence rules get named `rstest` cases in `cache-response/tests` built from plain key material, with no Python involved
