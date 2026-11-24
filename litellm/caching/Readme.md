# Caching on LiteLLM

LiteLLM supports multiple caching mechanisms. This allows users to choose the most suitable caching solution for their use case.

The following caching mechanisms are supported:

1. **RedisCache**
2. **RedisSemanticCache**
3. **QdrantSemanticCache**
4. **InMemoryCache**
5. **S3Cache**
6. **AzureBlobCache**
7. **DualCache** (updates both Redis and an in-memory cache simultaneously)

## Folder Structure

```
litellm/caching/
├── base_cache.py
├── caching.py
├── caching_handler.py
├── dual_cache.py
├── in_memory_cache.py
├── qdrant_semantic_cache.py
├── redis_cache.py
├── redis_semantic_cache.py
├── s3_cache.py
```

## Embeddings and shared Redis memory

The response cache stores embeddings per input item. A batched embeddings request with N inputs writes N keys, each holding a full vector. A batch job over unique texts gets almost no cache hits, yet it can fill a Redis instance. Once Redis reaches `maxmemory`, a `volatile-lru` policy evicts any key with a TTL, and that includes the proxy's rate limit counters, spend and budget state, and auth cache entries when they share the instance. A shorter cache TTL does not prevent this, since the cache can still fill memory before keys expire.

To stop caching embeddings while keeping every other default call type, exclude them:

```yaml
litellm_settings:
  cache: true
  cache_params:
    type: redis
    excluded_call_types: ["embedding", "aembedding"]
```

`excluded_call_types` removes names from `supported_call_types`, so list both the sync and async name. Listing every call type you do want in `supported_call_types` without the embedding names has the same effect.

To keep caching embeddings, give the response cache its own Redis instance. A different `namespace` or database number on the same instance does not help, because `maxmemory` and eviction apply to the whole instance. Point `general_settings.coordination_redis`, which holds rate limits, spend tracking and locks, and `router_settings` (`redis_host`, `redis_port`) at a different instance from `cache_params`:

```yaml
litellm_settings:
  cache: true
  cache_params:
    type: redis
    host: cache-redis
    port: 6379

general_settings:
  coordination_redis:
    host: coordination-redis
    port: 6379

router_settings:
  redis_host: coordination-redis
  redis_port: 6379
```

## Documentation
- [Caching on LiteLLM Gateway](https://docs.litellm.ai/docs/proxy/caching)
- [Caching on LiteLLM Python](https://docs.litellm.ai/docs/caching/all_caches)







