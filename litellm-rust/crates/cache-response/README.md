# Response cache foundation

`ResponseCache<B>` adds request keys, independent read/write controls, response envelopes, and freshness checks to any `B: BaseCache<Value = CacheEntry>`

## Ownership

`litellm-cache` defines typed storage and codec traits. Memory and Redis implement those traits without depending on response policy. Other consumers can store their own value types using the same backend implementations

`litellm-cache-response` owns response keys, controls, entries, the Python-compatible response codec, and `WriteBuffer`, the backend-neutral deferred-write policy. It has no runtime dependency on a specific cache backend or Python

The Python bridge constructs backends and selects them through its private `NativeResponseCache` enum, which only dispatches. Generic Rust callers inject their backend directly. A native gateway can construct the same generic response service in its own host

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

For Redis, inject `RedisCache::new(url, ttl, ResponseCacheCodec)` instead. Namespaces are optional and existing namespace prefixes are preserved. Sync operations check out independent connections from a bounded pool, while async callers, including counters and claims, move that blocking work off the executor. The pool skips the checkout PING and instead discards any connection whose command failed

Callers supply Unix time for response freshness. Backend TTL uses its own clock. A read can reject an entry through `max_age` even while the backend still retains it

## Python integration boundary

The extension keeps a private test harness for memory and Redis single and batch response lookup and storage. Batch lookup returns ordered values plus missing indices for embedding partial-hit wiring. No bridge-only cache type is part of the public API

Object responses are written as they are, and every other response shape is written as a serialized string, which is the pair of shapes Python reads. A string on the wire is therefore always a serialized response, so string-valued responses round trip. Typed backends such as memory never pass through the codec

The resolver reads the namespace's `cache` attribute each time it resolves. A captured binding retains the selected service for its operation, including background writes. `None` disables caching. Custom Python cache objects keep their original methods, arguments, returned awaitables, exceptions, and caller-task execution

Python callbacks use the built-in `Cache` API, so a `Cache` subclass works unchanged. A batch lookup takes one original kwargs mapping per request and returns the list of `get_cache` or gathered `async_get_cache` results, while native bindings return `{values, missing_indices}`. A batch store hands the caller's original result to `async_add_cache_pipeline`. `ping` calls `ping`, and a flush goes to the facade's backend

The private facade test harness checks object identity, method overrides, effective TTL, Redis namespace, memory capacity, and later configuration changes before selecting native execution. Its snapshot includes Redis connection settings, so a later `redis_kwargs` change, including an SSL option, selects Python callback execution. Buffered async writes honor `redis_flush_size`. Public activation must construct the shared native service from the initial Python Redis settings, including `litellm.default_redis_ttl` and SSL options. A buffered entry keeps the time it was produced, and a failed flush drops its batch instead of growing the buffer during an outage. The harness does not migrate entries or replace Python methods. Until activation configures one shared service, the Python facade and native test service can hold separate data. Existing public cache constructors remain on Python

Native cache handles must be recreated after fork. The bridge releases the GIL around native operations, and Redis runs blocking connection operations off the async executor. Native errors propagate to the host, which owns the existing fail-open and logging policy

The Redis backend also provides the primitives needed to preserve its direct Python surface later: TLS URLs, ping, bulk delete, counter batches, TTL, scan, set membership, raw queue push and pop, queue and counter pipelines, counter floor and maximum operations, script evaluation, client information, namespaced flush, and full flush. These are backend operations only and are not exported to Python by this PR. Memory provides TTL, oldest-key, and counter-pipeline operations

## Adding another backend

Implement `BaseCache` for the backend with its associated value type, and accept a `CacheCodec` when wire serialization is needed. `ResponseCache<B>` then works without another response implementation. Add a concrete bridge enum variant and constructor only when exposing that backend to Python

Verify typed values, TTL precedence, missing entries, serialization failures, namespaces, batch ordering, and sync/async behavior. Run response fixtures with `ResponseCacheCodec`, including both Python envelope encodings, before enabling a public facade

## Follow-up scope

Public SDK, Router, and proxy activation still need constructor parity, stream replay, embedding partial-batch integration, response reconstruction, callback scheduling, and failure-policy integration. This foundation does not switch those request paths

Redis cluster, disk, and cloud stores remain follow-ups. Semantic backends plug in through `SemanticCacheContext`, which carries the prompt inputs and metadata alongside the cache TTL. The generic dual cache takes read, write, and remote-failure policies, runs its async operations through the async L2 methods, and provides L2-first counters and atomic affinity claims. Errors propagate by default, and `RemoteFailurePolicy::UseLocal` opts key-value operations and claims into the local tier when L2 is unavailable. Claims compare decoded values, so a pin written by Python still matches. Public Router integration remains follow-up work. Reservations and pubsub still need explicit capabilities owned by their consuming features. Adding a cache backend does not establish those guarantees
