# Cache

## Python surface today

- `litellm.Cache` (`litellm/caching/caching.py`) is the response cache. It owns the cache key (`get_cache_key`), which call types are cached and TTL
- It wraps a backend in `self.cache`, a `BaseCache` such as `RedisCache`, `RedisClusterCache`, `S3Cache`, `GCSCache`, `AzureBlobCache`, `InMemoryCache` or a semantic cache
- `Cache` and `ResponseCache` are not the same thing. `BaseCache` and `DualCache` are also used as plain key-value state (cooldowns, spend queues, auth cache), which is not response caching
- Users can configure the cache or pass their own object

## Shared crates

- `crates/cache` mirrors Python's `BaseCache`, including its counter, claim and TTL capabilities. It keeps Python's shape on purpose
- `crates/cache-response` holds the one response-cache key builder (`cache_key`). Python's `Cache.get_cache_key` becomes a shim over it, never a third builder
- Backend crates (`cache-redis`, `cache-s3`, `cache-gcs`, `cache-azure-blob`, `cache-disk`, `cache-memory`, the semantic caches) implement the backends
- Redis-backed storage follows [redis](redis.md)

## Per host

- python-sdk and python-proxy
  - The core asks for the cache through a host call, and the configured Python object answers (`litellm/rust_bridge/host/cache.py`)
  - The built-in Python cache classes keep their API and become shims over the shared crates, so the answer may land in Rust again. That is fine because the logic exists once
  - A user subclass or custom cache is just another Python object answering the call
  - No shortcut that unwraps the native store inside the host driver. Speed on this path doesn't matter
- rust-proxy
  - Always the Rust cache, built from `config.yaml`. No other way

## Wire contracts

- Response-cache entries in Redis follow the Redis view rule in [redis](redis.md)
- Entries in other backends (S3, GCS, Azure Blob, disk): TBD

## Order and gates

- Python's `get_cache_key` moves onto `cache_response::cache_key` before Python and Rust pods share a response cache
- Order of backend moves: TBD
