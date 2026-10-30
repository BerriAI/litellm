"""Stand-in Router entry points for tests that mock the router with a MagicMock."""

from types import SimpleNamespace
from typing import TypeVar
from unittest.mock import MagicMock, NonCallableMock

from litellm.router_api import SETTING_FIELDS, ApplyResult, ChangeSet, RouterSettings

__all__ = ["route_entry_points_to_stubs"]

_RouterDouble = TypeVar("_RouterDouble")
_NOT_GIVEN = object()


def route_entry_points_to_stubs(router: _RouterDouble) -> _RouterDouble:
    """Make a MagicMock router answer its entry points (``registry()``, ``settings()``,
    ``health()``) from per-query stubs set on the mock itself.

    Tests stub one query by the name the router documents it under
    (``router.get_model_list = MagicMock(...)``, ``router.model_list = [...]``,
    ``router.get_configured_display_name = ...``) and every caller sees that stub,
    whichever entry point it reads through (``registry().deployments(...)``,
    ``registry().all()``, ``registry().group(name, field="display_name")``, ...). The
    stubs are looked up at call time, so they can be set or replaced after this call.
    The mapping is the one in litellm/router_deprecated.py.

    A hand-written fake router (a plain class defining only the old members it needs) works
    too: it is given mock entry points that read through to those members, and a member the
    fake does not define raises AttributeError on the call that needs it, as it did before.
    """
    for entry_point in ("registry", "health", "settings", "apply", "select"):
        if not isinstance(getattr(router, entry_point, None), NonCallableMock):
            object.__setattr__(router, entry_point, MagicMock())  # a frozen dataclass fake too
    registry = router.registry.return_value
    health = router.health.return_value

    def _given(**kwargs):
        # Pass on only the arguments a caller actually gave, the way the old call sites did.
        return {k: v for k, v in kwargs.items() if v is not None and v is not False}

    def _deployments(model_name=None, *, source="group", team_id=None, model_alias=None, request=None, call_type=None):
        if source == "group":
            return router.get_model_list(**_given(model_name=model_name, team_id=team_id))
        if source == "all":
            return router.get_all_deployments(model_name=model_name, **_given(model_alias=model_alias, team_id=team_id))
        if source == "config":
            return router.config_deployments()
        if source == "alias":
            return router.get_model_list_from_model_alias()
        if source == "routing_groups":
            return router.get_model_list_from_routing_groups(model_name=model_name)
        if source == "routed_group":
            return router.get_model_list_of_routed_group(model_name, **_given(team_id=team_id))
        if source == "routable":
            return router.get_routable_deployments(
                model_name, *(() if request is None else (request,)), **_given(call_type=call_type)
            )
        if source == "first_in_group":
            return router.get_deployment_by_model_group_name(model_group_name=model_name)
        if source == "named":
            return router.deployments_named(model_name)
        if source == "wildcard":
            return router.wildcard_route_deployments(model_name, team_id)
        if source == "for_request":
            return router.deployments_for_request(model_name, request or {})
        raise AssertionError(f"unstubbed deployments source {source!r}")

    def _names(model_name=None, *, kind="groups", team_id=None, exclude_team_models=False):
        if kind == "groups":
            return router.get_model_names(**_given(team_id=team_id))
        if kind == "group_set":
            return router.model_names
        if kind == "ids":
            return router.get_model_ids(**_given(model_name=model_name, exclude_team_models=exclude_team_models))
        if kind == "deployment_names":
            return router.deployment_names
        if kind == "team_public":
            return router.team_public_model_names
        if kind == "litellm_models":
            return router.resolved_litellm_models(model_name, **_given(team_id=team_id))
        if kind == "route_candidates":
            return router.get_candidate_model_ids_for_route(model_name, **_given(team_id=team_id))
        if kind == "wildcard":
            return router.wildcard_routed_models(model_name, team_id)
        raise AssertionError(f"unstubbed names kind {kind!r}")

    def _resolve(name, *, kind="group", team_id=None, custom_llm_provider=_NOT_GIVEN, served_model=None):
        if kind == "group":
            return router.routable_model_group(name)
        if kind == "alias":
            return router.get_model_from_alias(model=name)
        if kind == "served":
            return router.resolve_served_model_name(name, served_model)
        if kind == "team_model":
            return router.map_team_model(
                name, team_id, custom_llm_provider=None if custom_llm_provider is _NOT_GIVEN else custom_llm_provider
            )
        if kind == "recognized":
            return router.is_recognized_model(name)
        if kind == "upstream":
            return router.get_routable_upstream_model(name, **_given(team_id=team_id))
        if kind == "search_tool_support":
            return router.get_search_tool_support(name, **_given(team_id=team_id))
        if kind == "from_id":
            # The old call sites passed the provider positionally, exactly when the new ones pass it.
            if custom_llm_provider is _NOT_GIVEN:
                return router.resolve_model_name_from_model_id(name)
            return router.resolve_model_name_from_model_id(name, custom_llm_provider)
        if kind == "group_of_id":
            return router.get_model_group(id=name)
        if kind == "wildcard_access_group":
            return router.is_model_access_group_for_wildcard_route(model_access_group=name)
        if kind == "team_model_alternatives":
            return router.team_model_has_alternatives(name)
        if kind == "routing_group_alternatives":
            return router.routing_group_has_alternatives(name)
        raise AssertionError(f"unstubbed resolve kind {kind!r}")

    def _group(name, *, field="info", include_hidden=False, team_id=None):
        if field == "info":
            return router.get_model_group_info(model_group=name, **_given(team_id=team_id))
        if field == "display_name":
            return router.get_configured_display_name(name)
        if field == "service_tiers":
            return router.get_configured_service_tiers(name, **_given(team_id=team_id))
        if field == "listing_info":
            return router.get_model_listing_info(name)
        if field == "mode":
            return router.get_configured_mode(name)
        if field == "token_limits":
            return router.get_configured_token_limits(name)
        if field == "cached_info":
            return router.cached_model_group_info(name)
        raise AssertionError(f"unstubbed group field {field!r}")

    registry.deployments.side_effect = _deployments
    registry.names.side_effect = _names
    registry.resolve.side_effect = _resolve
    registry.group.side_effect = _group
    registry.all.side_effect = lambda: router.model_list
    registry.aliases.side_effect = lambda: router.model_group_alias
    registry.deployment.side_effect = lambda id: router.get_deployment(model_id=id)
    registry.deployment_dict.side_effect = lambda id: router.get_model_info(id=id)
    registry.has_deployment.side_effect = lambda id: router.has_model_id(id)
    registry.discovered_model_info.side_effect = lambda id: router.get_discovered_model_info(id)

    def _credentials(id, *, form="params", **team):
        # Old call sites passed team_id (even None) exactly when the new ones do, so pass it on as given.
        if form == "params":
            return router.get_deployment_credentials(model_id=id)
        if form == "with_provider":
            return router.get_deployment_credentials_with_provider(model_id=id, **team)
        if form == "deployment":
            return router.get_credential_deployment(model_id=id, **team)
        raise AssertionError(f"unstubbed credentials form {form!r}")

    registry.credentials.side_effect = _credentials
    registry.access_groups.side_effect = lambda **kwargs: router.get_model_access_groups(**_given(**kwargs))
    registry.plugins.side_effect = lambda kind: getattr(router, f"{kind}_routers")
    health.fully_blocked_model_names.side_effect = lambda: router.get_fully_blocked_model_names()
    health.split_priority.side_effect = lambda model, kwargs: router.split_request_priority(model=model, kwargs=kwargs)
    registry.index.side_effect = lambda name: getattr(router, f"{name}_to_deployment_indices")
    registry.patterns.side_effect = lambda team=False: router.team_pattern_routers if team else router.pattern_router
    health.is_blocked.side_effect = lambda deployment: router.is_deployment_blocked(deployment)
    health.all_blocked.side_effect = lambda deployments: router.are_all_deployments_blocked(deployments)
    settings_backend = SimpleNamespace(
        _core_get_settings=lambda: router.get_settings(),
        _core_get_available_guardrail=lambda name: router.get_available_guardrail(guardrail_name=name),
        _core_get_allowed_fails_from_policy=lambda exception: router.get_allowed_fails_from_policy(exception=exception),
    )
    router.settings.side_effect = lambda: RouterSettings(
        **{name: getattr(router, name) for name in SETTING_FIELDS}, _router=settings_backend
    )

    def _apply(changes: ChangeSet) -> ApplyResult:
        for name, value in changes.assign.items():
            setattr(router, name, value)
        if changes.settings:
            router.update_settings(**changes.settings)
        upserted = [router.upsert_deployment(deployment=d) for d in changes.upsert]
        deleted = [router.delete_deployment(id=i) for i in changes.delete]
        if changes.redis_cache is not None:
            router.update_redis_cache(cache=changes.redis_cache)
        return ApplyResult(version=0, upserted=upserted, deleted=deleted)

    router.apply.side_effect = _apply

    async def _select(model, request_kwargs=None, *, kind="pick", **extra):
        given = _given(**extra)
        if kind == "pick":
            return await router.async_get_available_deployment(
                model=model, request_kwargs=request_kwargs if request_kwargs is not None else {}, **given
            )
        if kind == "healthy":
            return await router.async_get_healthy_deployments(
                model=model, request_kwargs=request_kwargs if request_kwargs is not None else {}, **given
            )
        if kind == "pass_through":
            return router.get_available_deployment_for_pass_through(model=model, request_kwargs=request_kwargs, **given)
        raise AssertionError(f"unstubbed select kind {kind!r}")

    router.select.side_effect = _select
    return router
