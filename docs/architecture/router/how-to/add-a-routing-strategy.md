# Add a custom routing strategy

Use this guide when none of the built-in strategies picks deployments the way
you need, and you want to plug in your own without changing the router. You
write a `CustomRoutingStrategyBase` and register it with
`router.register_plugin()`.

## Know what you replace

`register_plugin` replaces the router's whole pick, not only the strategy at its
end. The router calls your method and serves the deployment it returns as is.
The pre-routing hook, the filters and the built-in strategy no longer run unless
you call them.

![The built-in pre-routing, filters and strategy, next to a registered plugin's
async_get_available_deployment that replaces them and can call router.select to
keep the filters](../img/add-a-routing-strategy-1.svg)

To keep cooldowns, team checks and the other filters, ask the router for the
filtered candidates with `select(kind="healthy")` and choose only among them, as
the example below does.

## Write the strategy

```python
from litellm import Router
from litellm.types.router import CustomRoutingStrategyBase


class LowestIdStrategy(CustomRoutingStrategyBase):
    def __init__(self, router: Router) -> None:
        self.router = router

    async def async_get_available_deployment(
        self, model, messages=None, input=None, specific_deployment=False, request_kwargs=None
    ):
        healthy = await self.router.select(
            model,
            request_kwargs,
            kind="healthy",
            messages=messages,
            input=input,
            specific_deployment=specific_deployment,
        )
        if isinstance(healthy, dict):  # the name resolved to one specific deployment
            return healthy
        return min(healthy, key=lambda d: d["model_info"]["id"])

    def get_available_deployment(
        self, model, messages=None, input=None, specific_deployment=False, request_kwargs=None
    ):
        raise NotImplementedError("this strategy serves the async path only")
```

The method returns one deployment dict, normally one from the model list. The
router also serves a deployment that the registry does not hold.
`select(kind="healthy")` raises the router's usual no-deployments error when
everything is filtered out, so you do not need to raise your own.

## Register it

```python
router = Router(model_list=[...])
router.register_plugin(LowestIdStrategy(router))
```

From here on, `acompletion`, `aembedding` and the other async endpoint methods
all route through it. `get_available_deployment` serves `router.completion()`
and the other sync methods, so implement it too if you use them.

## Check it

`router.select(model)` now returns your pick without calling the provider, which
makes the strategy easy to unit test with mock deployments. [The
tutorial](../tutorial/route-your-first-requests.md) shows how to build a Router
with `mock_response` deployments.

## When a plugin is the better fit

If you only need to narrow the candidates, or to send some requests to a
different model group, you do not need to replace the pick. A deployment-level
filter fits as a `CustomLogger` that implements `async_filter_deployments` and
is added to `litellm.callbacks`. The router calls it in its callback filter
stage. Choosing a model group per request is what the sub-routers in
`litellm/router_strategy/auto_router/` and its siblings do in the pre-routing
hook. Both approaches keep the rest of the pipeline intact, including the
built-in strategy. See [How a deployment is picked](../explanation/selection.md)
and [Strategies, routing groups and sub-routers](../explanation/strategies.md)
for where each stage runs.
