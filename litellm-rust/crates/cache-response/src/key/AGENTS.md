# Cache keys

- Python-backed storage never builds its key in Rust. It gets the key from the Python cache's `get_cache_key(**kwargs)`, and a native backend never asks Python for one
- Native and Python keys need not be byte-identical, but they follow the same precedence for presets, model and caching groups, files, namespaces, and semantic tenant scope
- Where the two paths disagree, the native key is the stricter one. A divergence may cost hit rate but must never return a response cached for a different request
- Native keys hash every parameter the host forwards, as if Python's `enable_caching_on_provider_specific_optional_params` were on. The host removes litellm-owned kwargs before they reach key material
- Key material has no per-field participation. A rule that drops parameters from the key is injected when `ResponseCache<B>` is constructed, never decided per request
- The target follows Python's `_get_model_param_value`: the model group (with caching groups already applied by the host) when routed through one, otherwise model, provider and API base. Credentials never enter the key
- Headers the caller forwards are key material, with names lowercased. Headers the route adds for authentication or provider defaults are not
- A request that `before_provider_request` changed skips the cache on both storage paths, so neither derivation sees the rewritten request
- A key is a value computed once per call and handed back to the service that produced it. It is never written back into the request, and no precomputed key or key input rides on `CacheOptions`
- A key built outside this crate, such as Python's `get_cache_key` result, is `CacheKey::External` and never passes through `CacheKey::derive`
- Python-backed storage accepts only `CacheScope::Shared`, because Python's key cannot carry a caller scope. A `CacheScope::Caller` request against it fails key resolution and skips the cache
- Changing the layout in `CacheKey::derive` is a key migration: bump `KEY_VERSION` with it
