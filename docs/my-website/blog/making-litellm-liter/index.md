---
slug: making-litellm-liter
title: "Making LiteLLM Liter"
date: 2026-01-31T10:00:00
authors:
  - name: Ryan Crabbe
description: "How moving auth out of FastAPI's Depends() and into ASGI middleware eliminated solve_dependencies overhead and improved proxy throughput by 15%."
tags: [performance]
hide_table_of_contents: false
---

import { BeforeFlow, AfterFlow } from '@site/src/components/AuthFlowComparison';

Over the past week we focused on reducing LiteLLM proxy's baseline overhead. Across a handful of targeted changes we got a 15% increase in throughput and cut mean latency by 16%, with p95 and p99 dropping close to 20%. The single biggest win came from rethinking how we handle authentication.

<div style={{display: 'flex', justifyContent: 'center'}}>

| Metric | Baseline | After | % Change |
|---|---|---|---|
| Requests / sec | 7,419.12 | 8,562.82 | +15.4% |
| Total time | 1.35s | 1.17s | -13.3% |
| Mean latency | 517.6ms | 432.0ms | -16.5% |
| Median (p50) | 544.2ms | 456.2ms | -16.2% |
| p95 latency | 923.0ms | 742.3ms | -19.6% |
| p99 latency | 1,097.2ms | 900.9ms | -17.9% |
| Max latency | 1,261.6ms | 992.2ms | -21.4% |

</div>

---

## Finding the Problem

I ran a pyinstrument profile against the proxy using a mock OpenAI provider. The benchmark fires 2000 requests at the proxy and directly at the provider, 20 times over, and compares the two so you can see exactly how much overhead the proxy is adding. 40K total requests. Looking at the flamegraph, something stood out. FastAPI's dependency injection resolver (`solve_dependencies`) was eating a huge chunk of time on every single request before the route handler even ran. About ~11s of that was the actual auth resolution, which is a lot on its own. But the other ~10s was DI machinery overhead. Pyinstrument couldn't attribute it to any child call because the time was being spent inside `solve_dependencies` itself, doing the work of walking the dependency tree, inspecting signatures, and wiring up sub-dependencies.

![pyinstrument flamegraph showing solve_dependencies taking 21.20s](./profile_before.png)

So I looked at the code. Every authenticated route was doing `Depends(user_api_key_auth)`, and `user_api_key_auth` looked like this:

```python
async def user_api_key_auth(
    request: Request,
    api_key: str = fastapi.Security(api_key_header),
    azure_api_key_header: str = fastapi.Security(azure_api_key_header),
    anthropic_api_key_header: Optional[str] = fastapi.Security(
        anthropic_api_key_header
    ),
    google_ai_studio_api_key_header: Optional[str] = fastapi.Security(
        google_ai_studio_api_key_header
    ),
    azure_apim_header: Optional[str] = fastapi.Security(azure_apim_header),
    custom_litellm_key_header: Optional[str] = fastapi.Security(
        custom_litellm_key_header
    ),
) -> UserAPIKeyAuth:
    """
    Parent function to authenticate user api key / jwt token.
    """
```

All this function is doing is grabbing every potential API key header from the request and injecting them in, then the function continues and builds and validates the `UserAPIKeyAuth` object from there. Not too crazy, right? So why is it taking so long? This is actually the standard, documented way to handle auth in FastAPI<sup>[1]</sup>. Using `Depends()` and `Security()` is what the framework recommends.

But if you dig into how this works under the hood basically when a request hits the route, FastAPI sees the `Depends(user_api_key_auth)` and then sees that `user_api_key_auth` itself has 6 more sub-dependencies that need to be resolved first. What FastAPI does is recursively walk the tree: "oh, this `Depends()` needs this `Security()`, which also needs this `Security()`, which...", a 7-node dependency tree to resolve, for things that don't even depend on each other. Then it finally passes everything to the route handler. This is where all that time in `solve_dependencies` was going, because it was doing this on every single request.

<BeforeFlow />

---

## The Fix

So initially my thinking was to try to keep this in house. Is there a way to reduce overhead here while still living within the way we use the FastAPI framework? I didn't see anything meaningful there other than reducing the overhead of the `user_api_key_auth` method itself, which is more than it should be but isn't a big improvement on its own.

That line of thinking didn't go anywhere. The better alternative was two things:

**First**, move all of the auth work out of FastAPI's dependency injection and into a separate, lighter middleware. The middleware extracts the headers directly, runs the auth check before FastAPI even gets involved, and caches the result on `request.state`:

```python
class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        try:
            # extract all provider headers directly from the request
            api_key = request.headers.get("Authorization") or ""
            # ... azure, anthropic, google, etc.

            # run auth and cache result on request.state
            result = await _user_api_key_auth_builder(request=request, api_key=api_key, ...)
            request.state.user_api_key_dict = result
        except (HTTPException, Exception):
            pass  # fallback: let the endpoint's Depends() handle it

        return await call_next(request)
```

If auth fails in the middleware, the exception is caught and the request still proceeds to the route handler. The endpoint's `Depends(user_api_key_auth)` runs the auth check again as a fallback and handles the 401 properly, so auth is never actually skipped.

**Second**, strip all 6 `Security()` parameters off of `user_api_key_auth` itself. This is what actually kills the `solve_dependencies` overhead. Before, even if the auth result was cached, FastAPI would still resolve the 7-node dependency tree on every request just because the function signature declared those sub-dependencies. After the change, the signature is just `(request: Request)`:

```python
async def user_api_key_auth(request: Request) -> UserAPIKeyAuth:
    # if middleware already ran, return cached result
    if hasattr(request.state, "user_api_key_dict"):
        return request.state.user_api_key_dict

    # fallback: extract headers directly (no Security() params)
    api_key = request.headers.get("Authorization") or ""
    azure_api_key = request.headers.get("api-key")
    # ...
```

The route handlers still do `Depends(user_api_key_auth)`, so nothing breaks downstream. But now `solve_dependencies` has a single leaf node to resolve instead of 7, and in the happy path it just reads from `request.state` and returns immediately.

![pyinstrument flamegraph after, solve_dependencies no longer a bottleneck](./profile_after.png)

<AfterFlow />

---

<sup>[1]</sup> https://fastapi.tiangolo.com/tutorial/security/first-steps/
