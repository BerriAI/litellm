# Move a caller off a deprecated Router member

Use this guide when you see
`DeprecationWarning: Router.<name> is deprecated; use Router.<replacement>`, or
when you want to remove an entry from the surface allowlist. The old members
still work, through `DeprecatedRouterMembers` in `litellm/router_deprecated.py`,
and will be removed once callers have moved.

![Moving off a deprecated Router member: the warning names the replacement, you
rewrite the call, run the tests, then router_surface_check.py reports clean or
an unused entry to remove with
--write](../img/move-off-deprecated-members-1.svg)

## 1. Find the replacement

The warning text names it. Without a warning at hand, search
`litellm/router_deprecated.py` for the old name: every member there calls
`_note(name, replacement)`. [Entry points](../reference/entry-points.md) has the
same mapping in table form.

To see which old names are still used at runtime, run the code path and inspect
the counter.

```python
from litellm.router_deprecated import DEPRECATED_ROUTER_MEMBER_USES

print(DEPRECATED_ROUTER_MEMBER_USES.most_common())
```

## 2. Rewrite the call

Most rewrites are mechanical.

| Before | After |
| --- | --- |
| `llm_router.model_list` | `llm_router.registry().all()` |
| `llm_router.get_model_list(model_name=m)` | `llm_router.registry().deployments(m)` |
| `llm_router.get_deployment(model_id=i)` | `llm_router.registry().deployment(i)` |
| `llm_router.get_model_names()` | `llm_router.registry().names()` |
| `llm_router.model_names` | `llm_router.registry().names(kind="group_set")` |
| `llm_router.get_model_group_info(m)` | `llm_router.registry().group(m)` |
| `llm_router.routing_strategy` | `llm_router.settings().routing_strategy` |
| `llm_router.get_settings()` | `llm_router.settings().as_dict()` |
| `llm_router.cache` | `llm_router.health().store` |
| `llm_router.cooldown_cache` | `llm_router.health().cooldowns` |
| `await llm_router.async_get_available_deployment(model=m, request_kwargs=k)` | `await llm_router.select(m, k)` |
| `llm_router.upsert_deployment(deployment=d)` | `llm_router.apply(ChangeSet(upsert=[d]))` |
| `llm_router.get_routing_group(name)` | `llm_router.registry().routing_group(name)` |
| `llm_router.discard()` | `llm_router.retire()` |

Return types are the same as before, so type checks keep passing. Where a query
is overloaded on `kind`, `source` or `field`, pass the literal directly so that
pyright chooses the right overload.

## 3. Check the surface

```bash
uv run python tests/code_coverage_tests/router_surface_check.py
```

The check fails in two cases. The first is a new member or `router_utils` import
used from outside the router, which you fix by using an entry point. The second
is a listed entry that nothing outside the router uses anymore, which means your
change removed its last use. In that case, shrink the allowlist in the same PR.

```bash
uv run python tests/code_coverage_tests/router_surface_check.py --write
git diff tests/code_coverage_tests/router_surface_allowlist.json
```

The diff should only remove lines. If `--write` adds an entry, a new outside use
slipped in, and the right fix is in the caller, not the allowlist.

## Inside the router family

Modules under `litellm/router_utils/`, `litellm/router_strategy/` and
`litellm/router_rust/` are the router's own. They do not trigger the warning,
and the surface check lists their uses separately (`members_router_family`).
Routing plugins should take a `RoutingContext` and read through it, so that they
stop depending on Router members at all.
