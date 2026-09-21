# Response cache foundation

`ResponseCache<B>` adds request keys, independent read/write controls, response envelopes, and freshness checks to any `B: BaseCache<Value = CacheEntry>`

## Ownership

`litellm-cache` defines typed storage and codec traits. Memory and Redis implement those traits without depending on response policy. Other consumers can store their own value types using the same backend implementations

`litellm-cache-response` owns response keys, controls, entries, and the Python-compatible response codec. It has no runtime dependency on a specific cache backend or Python

The Python bridge constructs backends and selects them through its private `NativeResponseCache` enum. Generic Rust callers inject their backend directly. A native gateway can construct the same generic response service in its own host

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

For Redis, inject `RedisCache::new(url, ttl, ResponseCacheCodec)` instead. Namespaces are optional and existing namespace prefixes are preserved. Sync operations check out independent connections from a bounded pool, while async callers move that blocking work off the executor

Callers supply Unix time for response freshness. Backend TTL uses its own clock. A read can reject an entry through `max_age` even while the backend still retains it

## Python integration boundary

The extension exposes `NativeCacheHandle`, `CacheResolver`, and captured `CacheBinding` objects for host integration. Memory and Redis handles support single and batch response lookup and storage. Batch lookup returns ordered values plus missing indices for embedding partial-hit wiring

The resolver reads the namespace's `cache` attribute each time it resolves. A captured binding retains the selected service for its operation, including background writes. `None` disables caching. Custom Python cache objects keep their original methods, arguments, returned awaitables, exceptions, and caller-task execution

Explicit facade registration checks object identity, method overrides, effective TTL, and configuration changes before selecting native execution. Redis defaults come from the Python settings snapshot, including `litellm.default_redis_ttl`, and buffered async writes honor `redis_flush_size`. Registration does not migrate entries or replace Python methods. Until activation configures one shared service, a registered facade and its native handle can hold separate data. Existing public cache constructors remain on Python

Native cache handles must be recreated after fork. The bridge releases the GIL around native operations, and Redis runs blocking connection operations off the async executor. Native errors propagate to the host, which owns the existing fail-open and logging policy

## Adding another backend

Implement `BaseCache` for the backend with its associated value type, and accept a `CacheCodec` when wire serialization is needed. `ResponseCache<B>` then works without another response implementation. Add a concrete bridge enum variant and constructor only when exposing that backend to Python

Verify typed values, TTL precedence, missing entries, serialization failures, namespaces, batch ordering, and sync/async behavior. Run response fixtures with `ResponseCacheCodec`, including both Python envelope encodings, before enabling a public facade

## Follow-up scope

Public SDK, Router, and proxy activation still need constructor parity, stream replay, embedding partial-batch integration, response reconstruction, callback scheduling, and failure-policy integration. This foundation does not switch those request paths

Redis cluster, disk, cloud stores, and semantic caching remain follow-ups. The generic dual cache now provides L2-first counters and atomic affinity claims with local fallback, but public Router integration remains follow-up work. Reservations, queues, and pubsub still need explicit capabilities owned by their consuming features. Adding a cache backend does not establish those guarantees
