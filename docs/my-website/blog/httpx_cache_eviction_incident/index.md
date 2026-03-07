---
slug: httpx-cache-eviction-incident
title: "Incident Report: Cache Eviction Closes In-Use httpx Clients"
date: 2026-02-27T10:00:00
authors:
  - name: Ryan Crabbe
    title: Performance Engineer, LiteLLM
    url: https://www.linkedin.com/in/ryan-crabbe-0b9687214
  - name: Ishaan Jaff
    title: "CTO, LiteLLM"
    url: https://www.linkedin.com/in/reffajnaahsi/
    image_url: https://pbs.twimg.com/profile_images/1613813310264340481/lz54oEiB_400x400.jpg
  - name: Krrish Dholakia
    title: "CEO, LiteLLM"
    url: https://www.linkedin.com/in/krish-d/
    image_url: https://pbs.twimg.com/profile_images/1298587542745358340/DZv3Oj-h_400x400.jpg
tags: [incident-report, caching, stability]
hide_table_of_contents: false
---

**Date:** February 27, 2026
**Duration:** ~6 days (Feb 21 merge -> Feb 27 fix)
**Severity:** High
**Status:** Resolved

> **Note:** This fix is available starting from LiteLLM `v1.81.14.rc.2` or higher.

## Summary

A change to improve Redis connection pool cleanup introduced a regression that closed **httpx clients** that were still actively being used by the proxy. The `LLMClientCache` (an in-memory TTL cache) stores both Redis clients *and* httpx clients under the same eviction policy. When a cache entry expired or was evicted, the new cleanup code called `aclose()`/`close()` on the evicted value which worked correctly for Redis clients, but destroyed httpx clients that other parts of the system still held references to and were actively using for LLM API calls.

**Impact:** Any proxy instance that hit the cache TTL (default 10 minutes) or capacity limit (200 entries) would have its httpx clients closed out from under it, causing requests to LLM providers to fail with connection errors.

---

## Background

`LLMClientCache` extends `InMemoryCache` and is used to cache SDK clients (OpenAI, Anthropic, etc.) to avoid re-creating them on every request. These clients are keyed by configuration + event loop ID. The cache has:

- **Max size:** 200 entries
- **Default TTL:** 10 minutes

When the cache is full or entries expire, `InMemoryCache.evict_cache()` calls `_remove_key()` to drop entries.

The cached values are a mix of:
- **Redis/async Redis clients** — owned exclusively by the cache, safe to close on eviction
- **httpx-backed SDK clients** (OpenAI, Anthropic, etc.) — shared references, still in use by router/model instances

---

## Root Cause

[PR #21717](https://github.com/BerriAI/litellm/pull/21717) overrode `_remove_key()` in `LLMClientCache` to close async clients on eviction:

<details>
<summary>Problematic code added in PR #21717</summary>

```python
class LLMClientCache(InMemoryCache):
    def _remove_key(self, key: str) -> None:
        value = self.cache_dict.get(key)
        super()._remove_key(key)
        if value is not None:
            close_fn = getattr(value, "aclose", None) or getattr(value, "close", None)
            if close_fn and asyncio.iscoroutinefunction(close_fn):
                try:
                    asyncio.get_running_loop().create_task(close_fn())
                except RuntimeError:
                    pass
            elif close_fn and callable(close_fn):
                try:
                    close_fn()
                except Exception:
                    pass
```

</details>

The intent was correct for Redis clients — prevent connection pool leaks when cached Redis clients expire. But `LLMClientCache` also stores httpx-backed SDK clients (e.g., `AsyncOpenAI`, `AsyncAnthropic`). These clients:

1. Have an `aclose()` method (inherited from httpx)
2. Are still held by references elsewhere in the codebase (router, model instances)
3. Were being closed without any check on whether they were still in use

So when the cache evicted an entry, it would call `aclose()` on an httpx client that was still being used for active LLM requests → closed transport → connection errors.

---

## The Fix

[PR #22247](https://github.com/BerriAI/litellm/pull/22247) removed the `_remove_key` override entirely:

<details>
<summary>The fix (PR #22247)</summary>

```diff
 class LLMClientCache(InMemoryCache):
-    def _remove_key(self, key: str) -> None:
-        """Close async clients before evicting them to prevent connection pool leaks."""
-        value = self.cache_dict.get(key)
-        super()._remove_key(key)
-        if value is not None:
-            close_fn = getattr(value, "aclose", None) or getattr(
-                value, "close", None
-            )
-            ...
-
     def update_cache_key_with_event_loop(self, key):
```

</details>

The eviction now simply drops the reference and lets Python's GC handle cleanup, which is safe because:
- httpx clients that are still referenced elsewhere stay alive
- Unreferenced clients get cleaned up by GC naturally

The other improvements from PR #21717 were kept:
- **`max_connections` respected for URL-based Redis configs**, previously silently dropped
- **`disconnect()` now closes both sync and async Redis clients**, sync client was previously leaked
- **Connection pool passthrough**, when a pool is provided with a URL config, it's used directly instead of creating a duplicate

---

## Remediation

| Action | Status | Code |
|--------|--------|------|
| Remove `_remove_key` override that closes shared clients on eviction | ✅ Done | [PR #22247](https://github.com/BerriAI/litellm/pull/22247) |
| Add e2e test: evicted client still usable (capacity) | ✅ Done | [PR #22313](https://github.com/BerriAI/litellm/pull/22313) |
| Add e2e test: expired client still usable (TTL) | ✅ Done | [PR #22313](https://github.com/BerriAI/litellm/pull/22313) |

The e2e tests go through `get_async_httpx_client()` the same code path the proxy uses in production and assert the client is still functional after eviction. These run in CI on every PR against `main`. If anyone modifies `LLMClientCache` eviction behavior, overrides `_remove_key`, or adds any form of client cleanup on eviction, these tests will fail regardless of the implementation approach.
