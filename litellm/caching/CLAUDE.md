# Caching (`litellm/caching/`)

Multiple cache backends (Redis, in-memory, S3, disk) + HTTP client cache.

## HTTP Client Cache Safety
- **Never close HTTP/SDK clients on cache eviction.** `LLMClientCache._remove_key()` must not call `close()`/`aclose()` on evicted clients — they may still be used by in-flight requests. Doing so causes `RuntimeError: Cannot send a request, as the client has been closed.` after the 1-hour TTL expires. Cleanup happens at shutdown via `close_litellm_async_clients()`.
