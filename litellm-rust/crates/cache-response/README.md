# Response cache

`ResponseCache<B>` adds request keys, independent read/write controls, response envelopes, and freshness checks to any `B: BaseCache<Value = CacheEntry>`

## Ownership

`litellm-cache` defines typed storage, codec, and capability traits. `BaseCache` is only get, set, TTL, and pipeline writes. Everything else is an optional capability a backend implements only where its Python class defines the method: `DisconnectCache`, `ConnectionCache` (`test_connection`), `PingCache`, `BatchCache`, `DeleteCache`, `FlushCache`, counters, queues, TTL, scan, and scripts. Memory, Redis, disk, S3, GCS, and Azure Blob implement those traits without depending on response policy, so other consumers can store their own value types in the same backends

Semantic backends (Redis, Valkey, Qdrant) are generic over their embedder and codec, and share one prompt and embedding contract from `litellm_cache::semantic`. They take a `SemanticCacheContext`, so `ResponseCache` drives them the same way it drives exact backends

`litellm-cache-response` owns response keys, controls, entries, the Python-compatible response codec, and `WriteBuffer`, the backend-neutral deferred-write policy. It has no runtime dependency on a specific cache backend or Python

`ExactResponseCache` is the object-safe view of a `ResponseCache` over an exact backend. `ConnectionProbe` is the object-safe `test_connection`, implemented only when the backend implements `ConnectionCache`, so a host holds one next to its `ExactResponseCache` and reports the operation as unsupported otherwise, as Python's `BaseCache` does. Lookup, store, batch, and flush never require it

## Native Rust use

```rust
use std::{sync::Arc, time::Duration};
use litellm_cache_memory::InMemoryCache;
use litellm_cache_response::{CacheKeyInput, ResponseCache, ResponseCacheRequest};
use serde_json::json;

let cache = ResponseCache::new(Arc::new(InMemoryCache::default()));
let request = ResponseCacheRequest::new(CacheKeyInput {
    preset: Some("example:key".into()),
    ..Default::default()
});
let now = Duration::from_secs(100);
cache.store(&request, json!({"answer": 7}), now)?;
assert_eq!(cache.async_lookup(&request, now).await?, Some(json!({"answer": 7})));
```

For Redis, inject `RedisCache::new(url, ttl, ResponseCacheCodec)` instead. Namespaces are optional and existing namespace prefixes are preserved

Callers supply Unix time for response freshness. Backend TTL uses its own clock. A read can reject an entry through `max_age` even while the backend still retains it

## Python integration

The bridge activates backends through the Rust catalog in `litellm/rust_bridge/catalog.py`. Every cache rule ships as `PYTHON_ONLY`, so SDK, Router, and proxy calls stay on Python and construct no native cache resources until a rule is changed

When a rule selects a backend, the Python `Cache` facade builds the native runtime from its own configuration and routes its storage calls (sync and async lookup and store, and pipelined batch store) to it. Stream replay, embedding partial-hit merging, response reconstruction, and callbacks stay in Python on top of that native store. The Python backend object remains for its direct API

Object responses are written as they are, and every other response shape is written as a serialized string, which is the pair of shapes Python reads. A string on the wire is therefore always a serialized response, so string-valued responses round trip. Typed backends such as memory never pass through the codec

Native cache handles must be recreated after fork. Native errors propagate to the host, which owns the existing fail-open and logging policy

## Adding another backend

Implement `BaseCache` for the backend with its associated value type and the capability traits its Python class supports, and accept a `CacheCodec` when wire serialization is needed. `ResponseCache<B>` then works without another response implementation

Run the `litellm-cache-testing` contract checks the backend's capabilities allow, and run response fixtures with `ResponseCacheCodec`, including both Python envelope encodings, before adding a catalog rule
