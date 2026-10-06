# Cache keys

Python-backed storage never builds its key in Rust. Key construction is coupled to storage. The storage that holds an entry is the only thing that derives its key. Python-backed storage gets its key from the Python cache's `get_cache_key(**kwargs)`. Native storage gets its key from `litellm-cache-response`. Never derive a key in Rust for Python-backed storage, and never ask Python for a native backend's key

The two paths do not need byte-identical keys. A native key may hash typed JSON, the API surface, and the caller scope under a versioned namespace, while Python hashes formatted kwargs. Both paths must use the same precedence for presets, model and caching groups, files, namespaces, and semantic tenant scope. When the two paths disagree, the native key must be the stricter one: a divergence may cost hit rate but must never return a response cached for a different request. Native keys hash every parameter the host forwards, so they always include provider-specific parameters, as if Python's `enable_caching_on_provider_specific_optional_params` were on. litellm-owned kwargs are removed by the host before they reach key material, not filtered here

A native key identifies the model the way Python's `_get_model_param_value` does: `CacheTarget::ModelGroup` when the host routed through a model group, `CacheTarget::Deployment` otherwise. The host resolves the group, applying caching groups first where it supports them, and passes it as `litellm_core::CallOptions::model_group`. Under a group, deployment URLs and credentials never enter the key, so every deployment of one group shares entries and two groups never do. Without a group, the model, provider and API base form the target, so two deployments that share a model name never share entries. Credentials never enter the key

Headers the caller forwards (`extra_headers`, and `provider_specific_header` on Messages) are key material, with names lowercased. Headers the route adds for authentication or provider defaults are not

A request that `before_provider_request` changed skips the cache on both storage paths. Core holds the key input in a `GuardedCache` across that hook and releases it only when the URL, body, and headers (compared case-insensitively, last value wins) are unchanged, so neither derivation sees the rewritten request

## Structure

Everything that builds native key material lives in this module: the target, the request parameters and forwarded headers, and the derivation. Routes in core only map their request fields onto `CacheKeyInput::forwarded`. The native keying layout stays in `CacheKey::derive` in `derive.rs`; changing it is a key migration, so bump `KEY_VERSION` with it

Make the wrong path impossible to write rather than documented

- `ResponseCacheService::key` is a required method with no default implementation. Each storage adapter chooses its derivation explicitly
- `ResponseCacheRequest` carries key material (`CacheKeyInput`, surface, scope) and never a key. Nothing upstream of the service fills in a key or a precomputed key input
- Native derivation (`CacheKey::derive`) is private to `litellm-cache-response`. Adapters outside the crate cannot call it
- Key material carries no per-field participation. If a native path ever needs to drop parameters from the key, the rule lives in this crate and is injected when `ResponseCache<B>` is constructed, never decided per request
- The resolved key travels as a `CacheKey` value passed to `lookup` and `store`. `CacheKeyInput::Preset` means only the caller's `preset_cache_key`; do not write a resolved key back into the request
- Python-backed storage accepts only `CacheScope::Shared`, because Python's key cannot carry a caller scope. Its `ResponseCacheConfig::supports_isolated_scope` is false, so `ScopedCache::new` rejects `Isolated` at construction instead of silently dropping it
- `litellm-cache-response` has no pyo3 dependency. The Python adapter in `python-bridge/src/cache/python` asks Python for the key and does not project kwargs into Rust key input

Do not reintroduce side channels such as a precomputed key input or key context on `CacheOptions` or `ScopedCache`. If a route needs a new input to affect the key, add it to the key material and teach both derivations about it

## Tests

Types catch the accidental mistakes. Tests catch drift and deliberate workarounds

- Delegation: a Python cache subclass whose `get_cache_key` returns a fixed key must be the key used for lookup, store, and the cache-key response header on native inference
- Strictness: for each parameter class and both provider opt-in values, a parameter that changes the Python key also changes the native key
- Native `CacheKey::derive` precedence rules get named `rstest` cases in `cache-response/tests` built from plain key material, with no Python involved
