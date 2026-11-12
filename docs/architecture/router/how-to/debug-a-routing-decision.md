# Find out why a request went where it did

Use this guide when a request landed on an unexpected deployment, fell back when
it should not have, or failed with "No deployments available for selected
model". Work from what the response tells you toward what the router saw.

![Debugging unexpected routing: read the headers, check how the name resolved or
list the candidates the filters left, then check cooldowns, blocks, team and
order, or the strategy inputs](../img/debug-a-routing-decision-1.svg)

## 1. Read the response headers

Through the proxy, every response says what happened.

| Header | Tells you |
| --- | --- |
| `x-litellm-model-id` | the deployment that answered (`model_info.id`) |
| `x-litellm-model-group` | the model group that answered, which differs from the requested one after a fallback |
| `x-litellm-model-api-base` | the deployment's endpoint |
| `x-litellm-attempted-retries` | how many retries ran |
| `x-litellm-attempted-fallbacks` | how many fallback hops ran |

From the SDK, `response._hidden_params["model_id"]` holds the deployment id.

## 2. Check how the name resolved

```python
registry = router.registry()
registry.resolve("gpt-4", kind="alias")        # alias target, or None
registry.resolve("gpt-4", kind="group")        # the group the name routes to
registry.resolve("gpt-4", kind="recognized")   # does the router know the name at all
registry.names("gpt-4", kind="route_candidates", team_id=team_id)  # deployment ids it may reach
```

A name that is not a model group can still route through an alias, a routing
group, a team public name, a wildcard such as `openai/*`, or
`default_deployment`. [How a name finds its
deployments](../explanation/names.md) has the order.

## 3. List the candidates the filters left

`select(kind="healthy")` runs the whole filter chain for a request without
calling anything. Pass the same `request_kwargs` the real request had, since the
team filter, the tag filter and the per-request settings read them.

```python
candidates = await router.select(
    "gpt-4o",
    {"metadata": {"user_api_key_team_id": team_id, "tags": ["prod"]}},
    kind="healthy",
    messages=messages,
)
print([d["model_info"]["id"] for d in candidates])

picked = await router.select("gpt-4o", {"metadata": {...}}, messages=messages)
```

The router does not log its filter stages. Its filters run in the order that
[How a deployment is picked](../explanation/selection.md) shows, so compare the
`kind="healthy"` result with the group's deployments and walk through the checks
below.

## 4. Check what removes a deployment

```python
health = router.health()
ids = list(registry.names("gpt-4o", kind="ids"))
await health.cooldowns.async_get_active_cooldowns(model_ids=ids, parent_otel_span=None)
health.is_blocked(registry.deployment("gpt-4o-eastus"))
```

![Why a deployment is missing from the candidates: a cooldown entry, then
model_info.blocked, another team_id, a lower order in the group, and finally the
pre-call checks](../img/debug-a-routing-decision-2.svg)

A cooldown entry records the failure that caused it. The rules for when a
failure counts are in [Retries, fallbacks and
cooldowns](../explanation/reliability.md). Deployments with a higher `order` get
traffic only once every lower-order deployment of the group has failed for the
request.

The error "No deployments available for selected model" is a
`RouterRateLimitError`. Its message includes the cooldown list and the seconds
until the soonest cooldown ends, and it says whether pre-call checks were on,
since they are a common cause.

## 5. Check the strategy's inputs

If the right deployments made it through and the pick still looks wrong, read
what the strategy reads. `router.settings().routing_strategy` says which
strategy ran, unless a routing group or a per-request `routing_strategy`
overrode it. For `simple-shuffle`, check `weight`, `rpm` and `tpm` on the
deployments and any per-team `weights` in key or team `router_settings`. For
usage-based strategies, `await router.health().usage("gpt-4o")` shows this
minute's counters, and remember that workers see each other's counters about a
second late (see [Shared state](../explanation/shared-state.md)).
