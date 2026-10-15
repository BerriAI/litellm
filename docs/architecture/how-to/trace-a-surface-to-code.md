# How to trace a product surface to the code behind it

Quadrant: how-to. Written against `main` at `13118948` (2026-10-15)

Use this when someone reports a behavior on a URL, a UI page or an SDK call and you need to find every piece of code that shapes it. It takes about ten minutes once you've done it twice

## For a proxy URL

1. Find the handler. Most routes are decorated, so search for the path literal

   ```bash
   rg -n '"/v1/messages"' litellm/proxy enterprise/litellm_enterprise
   ```

   If nothing matches, the route is probably under a router `prefix=`, a path parameter like `/{provider}/...`, or added with `add_api_route`. Search for the last path segment instead

2. Check whether the route is lazy. If its prefix is in `LAZY_FEATURES` (`litellm/proxy/_lazy_features.py:68-326`), it only exists after the first matching request. That matters for anything that inspects `app.routes` at startup, including the componentized gateway and backend trims

   ```bash
   rg -n -B3 '"/v1/messages"' litellm/proxy/_lazy_features.py
   ```

3. Check auth. Look for `Depends(user_api_key_auth)` in the decorator or signature. If it's missing, look in the handler body and in [permission-checks.md](/mnt/project-files/invariants/permission-checks.md) for the group it belongs to

4. Follow the hand-off. Data-plane handlers almost all call `ProxyBaseLLMRequestProcessing(...).base_process_llm_request(route_type=...)`. The `route_type` string is the key to the rest of the path. Search for it in `route_llm_request.py` to see which Router method it becomes, then in `router.py` to see which SDK function the Router calls

   ```bash
   rg -n '"aresponses"' litellm/proxy/route_llm_request.py
   rg -n 'self.aresponses|def aresponses' litellm/router.py
   ```

5. Step through the dispatch module, not `main.py`. `litellm.completion`, `responses`, `anthropic_messages`, `embedding` and `ocr` are rebound to `litellm/*/dispatch.py` at `litellm/__init__.py:1450-1495`. Check `litellm/rust_bridge/catalog.py` to see whether this route and provider can run in Rust

6. Find the provider code. `get_llm_provider` (`litellm/litellm_core_utils/get_llm_provider_logic.py:161`) resolves the provider. Then look for `_complete_<provider>` in `main.py` to see whether it uses `base_llm_http_handler` or a legacy handler. The transformation is in `litellm/llms/<provider>/<endpoint>/transformation.py`

7. List the hooks that will run. Every object in `litellm.callbacks` can act at pre-call, during-call, post-call and logging. For a given config, the quickest way to see them is `GET /callbacks/list`, or read `PROXY_HOOKS` in `litellm/proxy/hooks/__init__.py` plus the config's `callbacks`, `guardrails` and `success_callback`

8. Check which subsystems that flow reaches using [flow-subsystem-matrix.md](../reference/flow-subsystem-matrix.md), and the known gaps for each in [gap-register.md](../reference/gap-register.md)

## For an Admin UI page

1. Find the page under `ui/litellm-dashboard/src/app/` (dashboard pages are under `(dashboard)/`)
2. Find the network calls it makes. Nearly all go through exported functions in `ui/litellm-dashboard/src/components/networking.tsx`

   ```bash
   rg -n 'export const .*Call' ui/litellm-dashboard/src/components/networking.tsx | rg -i key
   ```

3. Each function names its backend path. Continue from step 1 of the proxy URL recipe with that path

## For an SDK call

1. Start at `litellm/__init__.py` and find which module exports the name. Dispatch modules win over `main.py` for the five rebound verbs
2. Follow the `@client` wrapper in `litellm/utils.py:1712` to see caching, logging and cost, then continue from step 6 above

## Keep what you find

If the trace shows the docs here are wrong, fix the page and update its "Verified on" line. If it shows a new intent-vs-reality gap, add a row to [gap-register.md](../reference/gap-register.md)
