---
name: response-cache
description: Change response-cache key construction, storage selection, or the cache service contract in litellm-rust, across cache-response, core caching, and the Python bridge cache adapters
---

# Response cache

Key construction is coupled to storage. The storage that holds an entry is the only thing that derives its key. Python-backed storage gets its key from the Python cache's `get_cache_key(**kwargs)`. Native storage gets its key from `litellm-cache-response`. Never derive a key in Rust for Python-backed storage, and never ask Python for a native backend's key

The two paths do not need byte-identical keys. A native key may hash typed JSON, the API surface, and the caller scope under a versioned namespace, while Python hashes formatted kwargs. Both paths must go through the same mechanism: the same participation sets (API parameters, litellm-owned parameters, the internal kwarg prefix, and the provider opt-in flag) and the same precedence for presets, model and caching groups, files, namespaces, and semantic tenant scope. A parameter that changes one path's key must change the other's

## Structure

Make the wrong path impossible to write rather than documented

- `ResponseCacheService::key` is a required method with no default implementation. Each storage adapter chooses its derivation explicitly
- `ResponseCacheRequest` carries key material (typed fields, transport, rewritten request, surface) and never a key. Nothing upstream of the service fills in a key or a precomputed key input
- Native derivation (`build`, the hash function, `KeyRules`, and the keying layout in `scope.rs`) is private to `litellm-cache-response`. Adapters outside the crate cannot call it
- `KeyRules` are injected when a native `ResponseCache<B>` is constructed: from config on the gateway, from Python's parameter lists once on the bridge. Never import Python modules per request to decide participation
- The resolved key travels as a `CacheKey` value passed to `lookup` and `store`. `preset` means only the caller's `preset_cache_key`; do not write a resolved key back into the request
- Python-backed storage accepts only `CacheScope::Shared`, because Python's key cannot carry a caller scope. Reject `Isolated` at construction instead of silently dropping it
- `litellm-cache-response` has no pyo3 dependency. The Python adapter in `python-bridge/src/cache/python` asks Python for the key and does not project kwargs into Rust key input

Do not reintroduce side channels such as a precomputed key input or key context on `CacheOptions` or `ScopedCache`. If a route needs a new input to affect the key, add it to the key material and teach both derivations about it

## Tests

Types catch the accidental mistakes. Tests catch drift and deliberate workarounds

- Delegation: a Python cache subclass whose `get_cache_key` returns a fixed key must be the key used for lookup, store, and the cache-key response header on native inference
- Participation parity: for each parameter class and both provider opt-in values, varying one parameter changes both the Python key and the native key, or neither. Assert that agreement, not byte equality
- Table source: the native participation table equals `ModelParamHelper._get_all_llm_api_params()`, `all_litellm_params`, and `INTERNAL_KWARG_PREFIX` as loaded from the installed `litellm`
- Native `build` precedence rules get named `rstest` cases in `cache-response/tests` built from plain key material, with no Python involved
