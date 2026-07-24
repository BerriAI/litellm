# Vertex / Gemini explicit context caching

Handles the `cachedContents` (explicit context cache) flow for Vertex AI, Vertex AI Beta, and Gemini (Google AI Studio). When a request marks a prefix with `cache_control` breakpoints instead of passing a flat `cached_content` id, the discovery path in `vertex_ai_context_caching.py` resolves the cachedContent id by hashing the messages and issuing a live `cachedContents` LIST.

## In-memory cache of resolved cache ids

That LIST is a control-plane round trip (p50 1552ms) paid on every request. `id_cache.py` caches the resolved cachedContent `name` in-memory so a warm request skips it. Off by default; enable with

```yaml
litellm_settings:
  enable_vertex_context_cache_id_caching: true
```

or from the SDK

```python
import litellm
litellm.enable_vertex_context_cache_id_caching = True
```

It covers both Vertex AI and Gemini/AI Studio; the p50 1552ms saving is measured on Vertex.

Correctness properties, enforced by `id_cache.py`:

- Each entry's ttl is derived from the backing cachedContent's real `expireTime` (minus a 5s safety margin), never a fixed span from insertion, so a cache hit can never outlive its backing cachedContent (which would fail downstream `generateContent` with NOT_FOUND). If `expireTime` is absent or unparseable the entry is not stored, degrading safely to always-LIST
- The key is namespaced by `(custom_llm_provider, vertex_project, vertex_location, api_base, credential-hash)`, so an id created under one provider/project/location/endpoint/credential is never served to another
- The cache is bounded and in-memory (no Redis), so each process holds its own copy; a cold process still LISTs and so still dedups against caches created by peers
- ttl bounds natural expiry, but a cachedContent can also be deleted/revoked server-side before then. When generateContent then rejects a warm-hit id (Vertex reports a deleted cache as HTTP 400 INVALID_ARGUMENT, "Invalid resource state for cache content <id>", not a 404), the completion handler invalidates that entry (`invalidate_cache_id`) by matching the served id in the error body, so the next request re-resolves via LIST/create instead of re-serving the dead id
