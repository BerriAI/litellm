# Change deployments or settings at runtime

Use this guide to add, replace or remove deployments, or to change router
settings, on a Router that is already serving traffic. All changes go through
`router.apply()` with a `ChangeSet`. Do not assign Router attributes or mutate
the lists that `settings()` and `registry()` return.

![Sequence in which apply(ChangeSet) runs assign, settings, upsert, delete and
redis_cache, bumps the registry version per deployment change, and returns
ApplyResult](../img/change-the-router-at-runtime-1.svg)

## Add or replace a deployment

`upsert` matches on `model_info.id`. A deployment with a new id is added, and
one with an existing id replaces the old one.

```python
from litellm.router_api import ChangeSet
from litellm.types.router import Deployment

result = router.apply(ChangeSet(upsert=[
    Deployment(
        model_name="gpt-4o",
        litellm_params={"model": "azure/gpt-4o-westus", "api_base": "...", "api_key": "..."},
        model_info={"id": "gpt-4o-westus"},
    )
]))
previous = result.upserted[0]  # the deployment it replaced, or None
```

## Remove a deployment

```python
result = router.apply(ChangeSet(delete=["gpt-4o-westus"]))
removed = result.deleted[0]  # the removed deployment, or None if the id was unknown
```

## Change settings

Use `settings` for the validated runtime settings that `update_settings`
accepted, and `assign` to replace a field from `SETTING_FIELDS` as it is.

```python
router.apply(ChangeSet(settings={"num_retries": 3, "routing_strategy": "latency-based-routing"}))
router.apply(ChangeSet(assign={"cooldown_time": 30, "fallbacks": [{"gpt-4o": ["claude-sonnet"]}]}))
```

`assign` takes names from `SETTING_FIELDS`, the plain settings in
`PLAIN_SETTING_ATTRIBUTES` (`num_retries`, `max_fallbacks`, `retry_after`,
`timeout`) and `model_group_alias`, and rejects anything else with a
`ValueError`. The values are copied, so changing your own dict afterwards does
not reach the router. `settings` is quieter: it skips keys outside
`RUNTIME_UPDATABLE_ROUTER_SETTINGS` with only a debug log, so check the name if
a change seems to do nothing. `settings` also casts the numeric values, rebuilds
routing groups when the strategy changes, and switches the strategy's loggers.
`assign` does none of that.

## Changes that need I/O

Re-reading discovered model info is async, so it needs `aapply`.

```python
await router.aapply(ChangeSet(refresh_model_info=True))
```

## Several changes at once

One `ChangeSet` can carry all of these fields. They are applied in the order
`assign`, `settings`, `upsert`, `delete`, `redis_cache`, and
`ApplyResult.version` is the registry version after the last one. The fields are
not swapped in as one, so a request that runs meanwhile can see part of the
change set, and an error part way through leaves the earlier fields applied. An
unknown `assign` name is refused before anything changes.

## Old code you may find

| Old | New |
| --- | --- |
| `router.upsert_deployment(d)` | `router.apply(ChangeSet(upsert=[d]))` |
| `router.delete_deployment(id=i)` | `router.apply(ChangeSet(delete=[i]))` |
| `router.update_settings(**s)` | `router.apply(ChangeSet(settings=s))` |
| `router.cooldown_time = 30` | `router.apply(ChangeSet(assign={"cooldown_time": 30}))` |
| `router.update_redis_cache(cache=c)` | `router.apply(ChangeSet(redis_cache=c))` |
| `await router.arefresh_model_info()` | `await router.aapply(ChangeSet(refresh_model_info=True))` |

The old forms still work through `DeprecatedRouterMembers` and warn. A write to
an old setting name becomes an `apply(ChangeSet(assign=...))` and takes effect
like one. Router settings are meant to be fixed after construction except
through `apply`, so new code should call `apply` directly.
