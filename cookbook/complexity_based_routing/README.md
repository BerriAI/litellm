# Complexity-based routing: send simple tasks local, complex tasks up-tier

If you're running a hybrid stack (local models + cloud fallback) and want most
traffic to stay on the cheap/local tier, escalating only when a task genuinely
needs it, LiteLLM's built-in routing strategies (`latency-based-routing`,
`cost-based-routing`, `usage-based-routing`, etc.) don't quite cover this: they
route based on deployment health/cost, not on what the *request itself* needs.

This recipe uses LiteLLM's `CustomRoutingStrategyBase` extension point to add a
fourth axis: **task complexity**. One `model_name` (`task/auto`) fronts a tier
ladder; a cheap classifier picks which underlying deployment actually serves
each request.

Built and validated on a real local+cloud hybrid stack (a 9B model on a
consumer GPU as the "standard" tier, a 4B model on a second GPU as the "simple"
tier, cloud as last resort) — not just a toy example. Two real findings worth
calling out are below.

## The tier ladder

```
task/auto
  ├─ simple    -> small/fast local model      (lookups, formatting, status checks)
  ├─ standard  -> primary local model         (default -- everyday agentic work)
  ├─ deep      -> primary local model, again  (same model, reasoning/thinking mode ON)
  └─ cloud     -> cheap cloud model           (only for things local genuinely can't do)
```

The `deep` tier is the one easy to miss: if your primary local model supports a
toggleable reasoning/thinking mode (Qwen3, DeepSeek-R1-distills, etc.), register
it as **two separate `model_name` entries** pointing at the same underlying
model/deployment — one with thinking off (fast, default), one with thinking on
(slow, free). That gives you a "local deep reasoning" rung between "fast local"
and "cloud," and in our testing it covered a lot of ground that looked like it
needed cloud but actually just needed more thinking time:

```yaml
model_list:
  - model_name: primary-model
    litellm_params:
      model: ollama_chat/your-model
      api_base: http://localhost:11434
      think: false        # fast, default tier

  - model_name: primary-model-deep
    litellm_params:
      model: ollama_chat/your-model   # same model
      api_base: http://localhost:11434 # same deployment
      think: true          # reasoning on -- still $0, just slower
      timeout: 600          # give it room; reasoning can take tens of seconds
```

## The classifier

Kept deliberately cheap: a heuristic gate first (keyword/length signals, near
zero cost), falling back to one small-model classification call only for the
ambiguous middle. Classify once per task and cache the decision — re-scoring
every turn of a conversation risks flip-flopping models mid-thread, which is
worse than picking one tier and staying with it.

```python
from litellm.router import CustomRoutingStrategyBase

TIER_MODEL_NAMES = {
    "simple": "small-model",
    "standard": "primary-model",
    "deep": "primary-model-deep",
    "cloud": "cloud-model",
}

class ComplexityRoutingStrategy(CustomRoutingStrategyBase):
    def __init__(self, router, default_sync, default_async):
        self.router = router
        self._default_sync = default_sync
        self._default_async = default_async

    async def async_get_available_deployment(self, model, messages=None, **kw):
        if model != "task/auto":
            # not our model group -- delegate to the router's normal behavior
            return await self._default_async(model=model, messages=messages, **kw)

        tier = await classify(messages)  # heuristic, then LLM fallback -- your logic
        target = TIER_MODEL_NAMES[tier]
        for deployment in self.router.model_list:
            if deployment.get("model_name") == target:
                return deployment
        return None  # or fall back to a safe default tier
```

See `complexity_router.py` in this folder for the full version (heuristic
gate, classifier prompt, per-task caching, fallback-on-error).

## Two things that will cost you real debugging time if you don't know them going in

**1. `set_custom_routing_strategy` only intercepts what you tell it to.** It's
a plain `setattr` swap of `get_available_deployment` /
`async_get_available_deployment` on the router instance — it does not filter by
`model` for you. If your strategy doesn't explicitly pass through every
`model` name that *isn't* your custom group to the router's original method
(captured *before* you call `set_custom_routing_strategy`), every other model
in your config silently starts going through your custom logic too, including
ones with multiple load-balanced deployments that expect the built-in
strategy's behavior.

**2. `LITELLM_WORKER_STARTUP_HOOKS` fires before `llm_router` exists, and if
your hook function is a blocking `async def` that the proxy awaits directly,
it can hang your entire startup.** In our testing, the hook ran early enough
that `litellm.proxy.proxy_server.llm_router` was still `None`, and awaiting a
retry-loop inside the hook itself blocked the proxy from ever reaching
"Application startup complete" — a self-deadlock, since router construction
happens *after* the startup-hooks step completes. Fix: make the hook function
itself synchronous and fire-and-forget the actual wait/registration logic via
`asyncio.create_task(...)`, so the hook returns immediately and the polling
runs once the event loop is free (which, in practice, was ~0 seconds after
startup actually finished):

```python
def register():
    import asyncio
    asyncio.create_task(_wait_and_register())  # don't await this directly

async def _wait_and_register():
    import litellm.proxy.proxy_server as proxy_server
    for _ in range(180):
        if proxy_server.llm_router is not None:
            break
        await asyncio.sleep(1)
    else:
        return
    router = proxy_server.llm_router
    strategy = ComplexityRoutingStrategy(
        router, router.get_available_deployment, router.async_get_available_deployment
    )
    router.set_custom_routing_strategy(strategy)
```

Wire it up with:

```yaml
# model_list needs a placeholder entry for the group name to be recognized:
model_list:
  - model_name: task/auto
    litellm_params:
      model: ollama_chat/your-model   # never actually used at request time --
      api_base: http://localhost:11434 # the strategy always overrides the deployment
```

```bash
LITELLM_WORKER_STARTUP_HOOKS=complexity_router:register
```

(mount `complexity_router.py` into the container alongside your `config.yaml`)

## Validated results

Two real A/B findings from our stack that motivated the `deep` tier
specifically:

- A 4B model asked a multi-source comparison question returned a coherent
  answer with **every citation as a literal placeholder string instead of a
  real reference** -- a reproducible failure, not a one-off. The same question
  against the 9B model (still fully local) returned 46 real, correctly
  numbered citations with working URLs, in less time.
- A trivial one-word-answer prompt took 38-98 seconds against our primary
  model with no other config change -- turned out to be unsuppressed
  reasoning/thinking output on a model that supports toggling it, not routing
  overhead. Once we split that into two tiers (`think: false` fast lane by
  default, `think: true` as its own explicit tier), the fast lane dropped to
  ~0.4s and the reasoning tier became a deliberate, cheap escalation step
  instead of an accidental one.
