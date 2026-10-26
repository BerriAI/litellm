# Python v2 cache

Keep `litellm._v2.cache.Cache` import-compatible when reorganizing this package. This package owns Python cache factories and the adapter between the existing `BaseCache` interface and `NativeCacheHandle`

Construct native handles at this boundary and inject the adapter through the existing cache facade's `_backend` parameter. Keep the facade's `type`, namespace, and TTL consistent with the configured native backend

Keep storage implementation in the Rust storage crates and response-cache policy in `litellm-cache-response` and core. Do not duplicate cache-key generation, freshness rules, response encoding, or inference orchestration here

Preserve synchronous and asynchronous cache operations, including TTL forwarding and lifecycle methods. Validate Python values before passing them to typed native interfaces. Keep native extension imports lazy so importing the package does not require loading the extension

Extend the existing v2 cache tests in `tests/test_litellm_rust/test_v2.py` for behavioral changes, following that directory's `AGENTS.md`. Test observable cache behavior rather than package layout or implementation structure
