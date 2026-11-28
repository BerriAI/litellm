# cache-control-injection

A deployment's `cache_control_injection_points`, the prompt cache breakpoints LiteLLM adds to a call. Python's `litellm/integrations/anthropic_cache_control_hook.py`

- `point.rs` is the setting as a config spells it. It is LiteLLM's own vocabulary and no provider sees it, so it is neither a wire type of one format nor a routing setting. Only its `control` is wire data, the format's own `CacheControl`
- One module per API format places the points on a request of that format, and `messages.rs` is the only one so far. A new format reads the same points in a module of its own, never in the route or a provider adapter
- Each module is plain data in three steps: count what the request already carries, plan which points are placed, rebuild the request. It reads no cost map, no provider and no host, so a route and `admit` ask it the same question and a Python binding can wrap it as it is
- `served` in a format's module is the one place that says which points that format places. `admit` asks it for each deployment, so a point a module learns to place stops going to Python in the same change
- The test table in `messages.rs` holds each case to what Python's `maybe_inject_cache_control` sends, and `test_messages_points_match_the_rust_route` in `tests/unit/integrations/test_anthropic_cache_control_hook.py` holds Python to the same cases. Change both together
