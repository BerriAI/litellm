"""
DeprecatedRouterMembers: the Router members that outside code used before the entry points.

``litellm.Router`` mixes this class in so existing callers keep working while they move to the
entry points (``registry()``, ``settings()``, ``health()``, ``select()``, ``apply()``,
``observe()``, ``register_plugin()``, ``retire()``; see litellm/router_api.py). Every member here
returns what it returned before, with the type it had before, and names its replacement in a
``DeprecationWarning``. Reads go to the same Router implementation (``Router._core_<name>``) the
entry point uses; writes go through ``apply`` / ``aapply`` so the registry version moves.

Access from the router itself and from its own modules (``litellm.router*``) is not counted:
those are the router's internals and move behind the core interface separately. Every other access
is counted in ``DEPRECATED_ROUTER_MEMBER_USES`` so the remaining users show up before this class is
deleted (planned about a month after the core swap).
"""

# pyright: reportPrivateUsage=false, reportMissingTypeArgument=false, reportUnknownParameterType=false
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
# pyright: reportUnknownLambdaType=false
# This shim is part of the router (it reads Router's ``_core_*`` members, as Router's own methods do)
# and repeats each old member's signature exactly, untyped parts included (``dict``, ``list``), so
# callers keep the types they had. Typing those parts is the core's job, not the shim's.

import contextvars
import functools
import inspect
import sys
import warnings
from collections import Counter
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Generic, TypeVar, overload

from litellm.router_api import SETTING_FIELDS, ChangeSet

if TYPE_CHECKING:
    from typing import Any  # noqa: TID251  # only to repeat two old signatures that had Any, unchanged

    from opentelemetry.trace import Span

    from litellm.caching.caching import RedisCache
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLogging
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
    from litellm.router import Router
    from litellm.router_strategy.budget_limiter import RouterBudgetLimiting
    from litellm.router_utils.kubernetes_pod_discovery import KubernetesPodDiscovery
    from litellm.router_utils.ptu_shares import PTUTeamCeiling
    from litellm.types.router import (
        CustomRoutingStrategyBase,
        Deployment,
        DeploymentModelListingInfo,
        DeploymentTypedDict,
        GuardrailTypedDict,
        ModelGroupInfo,
        PreRoutingHookResponse,
        SearchToolTypedDict,
    )
    from litellm.types.utils import ModelInfo as ModelMapInfo

_T: Final = TypeVar("_T")

DEPRECATED_ROUTER_MEMBER_USES: Final[Counter[str]] = Counter()  # mutable-ok: the live tally _note increments

_INTERNAL_PREFIX: Final = "litellm.router"


# Router's own attribute hooks: a write or delete through an old name passes through them first.
_ATTRIBUTE_HOOKS: Final = frozenset({"__setattr__", "__delattr__"})


def _note(name: str, replacement: str) -> None:
    depth = 2  # rebind-ok: grows by one per attribute hook frame below
    caller = sys._getframe(depth)  # rebind-ok: follows depth up past those hooks
    while caller.f_code.co_name in _ATTRIBUTE_HOOKS and caller.f_globals.get("__name__") == "litellm.router":
        depth += 1
        caller = sys._getframe(depth)
    module: Final = caller.f_globals.get("__name__")
    if isinstance(module, str) and module.startswith(_INTERNAL_PREFIX):
        return
    DEPRECATED_ROUTER_MEMBER_USES[name] += 1
    warnings.warn(
        f"Router.{name} is deprecated; use Router.{replacement}",
        DeprecationWarning,
        stacklevel=depth + 1,
    )


def _as_router(obj: "DeprecatedRouterMembers") -> "Router":
    """The Router this mixin is part of (it is mixed into ``litellm.Router`` only)."""
    from litellm.router import Router

    if isinstance(obj, Router):
        return obj
    raise TypeError(f"{type(obj).__name__} is not a litellm.Router; DeprecatedRouterMembers is mixed into Router only")


def _public_attribute_error(obj: object, core_name: str, public_name: str, error: AttributeError) -> AttributeError:
    """
    ``error`` renamed to the public attribute when it is the router's own ``core_name`` that is
    missing, so the message is the one reading the old attribute raised; any other error unchanged.
    """
    if error.name not in (core_name, public_name) or error.obj is not obj:
        return error
    return AttributeError(f"'{type(obj).__name__}' object has no attribute '{public_name}'", name=public_name, obj=obj)


class _DeprecatedAttribute(Generic[_T]):
    """A former Router attribute, read through an entry point. Assignment still replaces it."""

    def __init__(self, replacement: str, read: Callable[["Router"], _T]) -> None:
        self.replacement = replacement
        self.read = read
        self.name = ""

    def __set_name__(self, owner: type, name: str) -> None:
        # An alias (``_routing_groups = routing_groups``) keeps the first name.
        self.name = self.name or name

    @overload
    def __get__(self, obj: None, owner: type) -> object: ...

    @overload
    def __get__(self, obj: "DeprecatedRouterMembers", owner: type) -> _T: ...

    def __get__(self, obj: "DeprecatedRouterMembers | None", owner: type) -> object:
        if obj is None:
            return getattr(owner, "_core_" + self.name, self)
        _note(self.name, self.replacement)
        router: Final = _as_router(obj)
        try:
            return self.read(router)
        except AttributeError as e:
            raise _public_attribute_error(router, "_core_" + self.name, self.name, e) from None

    def __set__(self, obj: "DeprecatedRouterMembers", value: _T) -> None:
        _note(self.name, self.replacement)
        if self.name in SETTING_FIELDS:
            _as_router(obj).apply(ChangeSet(assign={self.name: value}))
        else:
            setattr(obj, "_core_" + self.name, value)

    def __delete__(self, obj: "DeprecatedRouterMembers") -> None:
        _note(self.name, self.replacement)
        try:
            delattr(obj, "_core_" + self.name)
        except AttributeError:
            # The only attribute this deletes; Python's own message for it, which sets no name/obj.
            raise AttributeError(f"'{type(obj).__name__}' object has no attribute '{self.name}'") from None


def _setting(name: str, read: Callable[["Router"], _T]) -> _DeprecatedAttribute[_T]:
    """A former settings attribute: ``read`` returns the router's own value, typed as before."""

    def read_setting(router: "Router") -> _T:
        if not router.settings().is_set(name):
            raise AttributeError(f"'{type(router).__name__}' object has no attribute {name!r}", name=name, obj=router)
        return read(router)

    return _DeprecatedAttribute(f"settings().{name}", read_setting)


def _attr(replacement: str, read: Callable[["Router"], _T]) -> _DeprecatedAttribute[_T]:
    return _DeprecatedAttribute(replacement, read)


# (id(router), member name) pairs whose subclass override is running in this context.
_OVERRIDES_RUNNING: Final[contextvars.ContextVar[frozenset[tuple[int, str]]]] = contextvars.ContextVar(
    "router_overrides_running", default=frozenset()
)


def _enter_override(name: str, override: Callable[..., object]) -> Callable[..., object]:
    """
    A direct call of a subclass override (``router.<name>()``) marks it running, so its
    ``super().<name>()`` reaches the parent once instead of re-entering the override through
    ``_core_<name>``.
    """
    if inspect.iscoroutinefunction(override):
        async_override: Final[Callable[..., Awaitable[object]]] = override

        @functools.wraps(override)
        async def entered_async(
            router: object,
            *args: object,
            **kwargs: object,  # kwargs-ok: passes the override's arguments through unchanged
        ) -> object:
            token: Final = _OVERRIDES_RUNNING.set(_OVERRIDES_RUNNING.get() | {(id(router), name)})
            try:
                return await async_override(router, *args, **kwargs)
            finally:
                _OVERRIDES_RUNNING.reset(token)

        return entered_async

    @functools.wraps(override)
    def entered(
        router: object,
        *args: object,
        **kwargs: object,  # kwargs-ok: passes the override's arguments through unchanged
    ) -> object:
        token: Final = _OVERRIDES_RUNNING.set(_OVERRIDES_RUNNING.get() | {(id(router), name)})
        try:
            return override(router, *args, **kwargs)
        finally:
            _OVERRIDES_RUNNING.reset(token)

    return entered


def _private_alias(name: str, member: Callable[..., object]) -> Callable[..., object]:
    """
    A private alias of an old member (``_get_all_deployments``). It was bound to Router's own
    method, so it never reached a subclass override of ``name``, and still doesn't.
    """
    return _enter_override(name, member)


def _route_override(cls: type, name: str, override: Callable[..., object]) -> Callable[..., object]:
    """
    Make the router's own calls to ``_core_<name>`` reach a subclass override of ``<name>``.

    When the override calls ``super().<name>()``, the call comes back to ``_core_<name>``; it is
    then sent on to the parent's implementation instead of back into the override.
    """
    core: Final = "_core_" + name

    def parent(router: object) -> Callable[..., object]:
        return getattr(super(cls, router), core)

    if inspect.iscoroutinefunction(override):
        async_override: Final[Callable[..., Awaitable[object]]] = override

        @functools.wraps(override)
        async def routed_async(
            router: object,
            *args: object,
            **kwargs: object,  # kwargs-ok: passes the override's arguments through unchanged
        ) -> object:
            key: Final = (id(router), name)
            running: Final = _OVERRIDES_RUNNING.get()
            if key in running:
                parent_async: Final[Callable[..., Awaitable[object]]] = getattr(super(cls, router), core)
                return await parent_async(*args, **kwargs)
            token: Final = _OVERRIDES_RUNNING.set(running | {key})
            try:
                return await async_override(router, *args, **kwargs)
            finally:
                _OVERRIDES_RUNNING.reset(token)

        return routed_async

    @functools.wraps(override)
    def routed(
        router: object,
        *args: object,
        **kwargs: object,  # kwargs-ok: passes the override's arguments through unchanged
    ) -> object:
        key: Final = (id(router), name)
        running: Final = _OVERRIDES_RUNNING.get()
        if key in running:
            return parent(router)(*args, **kwargs)
        token: Final = _OVERRIDES_RUNNING.set(running | {key})
        try:
            return override(router, *args, **kwargs)
        finally:
            _OVERRIDES_RUNNING.reset(token)

    return routed


class DeprecatedRouterMembers:
    """Old Router members, kept for about a month. Each names its replacement."""

    def __init_subclass__(cls, **kwargs: object) -> None:  # kwargs-ok: class keywords, passed on to super()
        # A Router subclass that overrides an old member (say async_pre_routing_hook) keeps
        # having its override called by the router, which now calls _core_<name> internally.
        super().__init_subclass__(**kwargs)
        for name, value in list(vars(cls).items()):
            if (
                name.startswith("_core_")
                or name not in vars(DeprecatedRouterMembers)
                or not hasattr(cls, "_core_" + name)
            ):
                continue
            if isinstance(value, staticmethod):
                # Its super() call reaches the old member here, which calls Router's own implementation.
                setattr(cls, "_core_" + name, value)
            elif inspect.isfunction(value):
                setattr(cls, "_core_" + name, _route_override(cls, name, value))
                setattr(cls, name, _enter_override(name, value))

    # -- registry: attributes --------------------------------------------------------------------
    model_list = _attr("registry().all()", lambda r: r._core_model_list)
    model_names = _attr("registry().names(kind='group_set')", lambda r: r._core_model_names)
    deployment_names = _attr("registry().names(kind='deployment_names')", lambda r: r._core_deployment_names)
    team_public_model_names = _attr("registry().names(kind='team_public')", lambda r: r._core_team_public_model_names)
    model_name_to_deployment_indices = _attr(
        "registry().index('model_name')", lambda r: r._core_model_name_to_deployment_indices
    )
    team_model_to_deployment_indices = _attr(
        "registry().index('team_model')", lambda r: r._core_team_model_to_deployment_indices
    )
    cached_deployment_model_info = _attr(
        "registry().deployment_model_info(id, model_name, cached=True)",
        lambda r: r._core_cached_deployment_model_info,
    )
    model_group_alias = _attr("registry().aliases()", lambda r: r._core_model_group_alias)
    pattern_router = _attr("registry().patterns()", lambda r: r._core_pattern_router)
    team_pattern_routers = _attr("registry().patterns(team=True)", lambda r: r._core_team_pattern_routers)
    routing_groups = _attr("registry().routing_groups()", lambda r: r._core_routing_groups)
    _routing_groups = routing_groups
    auto_routers = _attr("registry().plugins('auto')", lambda r: r._core_auto_routers)
    complexity_routers = _attr("registry().plugins('complexity')", lambda r: r._core_complexity_routers)
    quality_routers = _attr("registry().plugins('quality')", lambda r: r._core_quality_routers)
    adaptive_routers = _attr("registry().plugins('adaptive')", lambda r: r._core_adaptive_routers)
    routing_plugins = _attr("registry().plugins('routing')", lambda r: r._core_routing_plugins)
    prompt_caching_pin_cache = _attr(
        "registry().plugins('prompt_caching_pins')", lambda r: r._core_prompt_caching_pin_cache
    )

    # -- health and selection: attributes --------------------------------------------------------
    cache = _attr("health().store", lambda r: r._core_cache)
    cooldown_cache = _attr("health().cooldowns", lambda r: r._core_cooldown_cache)
    health_state_cache = _attr("health().health_checks", lambda r: r._core_health_state_cache)
    deployment_latency_map = _attr("health().latency", lambda r: r._core_deployment_latency_map)
    async_get_available_deployment = _attr(
        "select(model, request_kwargs)",
        lambda r: r._core_async_get_available_deployment,
    )

    # -- observers ------------------------------------------------------------------------------
    slack_alerting_logger = _attr("observe().slack_alerting_logger", lambda r: r._core_slack_alerting_logger)

    # -- settings -------------------------------------------------------------------------------
    routing_strategy = _setting("routing_strategy", lambda r: r._core_routing_strategy)
    context_window_fallbacks = _setting("context_window_fallbacks", lambda r: r._core_context_window_fallbacks)
    content_policy_fallbacks = _setting("content_policy_fallbacks", lambda r: r._core_content_policy_fallbacks)
    default_deployment = _setting("default_deployment", lambda r: r._core_default_deployment)
    default_litellm_params = _setting("default_litellm_params", lambda r: r._core_default_litellm_params)
    default_max_parallel_requests = _setting(
        "default_max_parallel_requests", lambda r: r._core_default_max_parallel_requests
    )
    cooldown_time = _setting("cooldown_time", lambda r: r._core_cooldown_time)
    allowed_fails = _setting("allowed_fails", lambda r: r._core_allowed_fails)
    allowed_fails_policy = _setting("allowed_fails_policy", lambda r: r._core_allowed_fails_policy)
    disable_cooldowns = _setting("disable_cooldowns", lambda r: r._core_disable_cooldowns)
    enable_pre_call_checks = _setting("enable_pre_call_checks", lambda r: r._core_enable_pre_call_checks)
    enable_health_check_routing = _setting("enable_health_check_routing", lambda r: r._core_enable_health_check_routing)
    health_check_ignore_transient_errors = _setting(
        "health_check_ignore_transient_errors", lambda r: r._core_health_check_ignore_transient_errors
    )
    background_health_check_model_groups = _setting(
        "background_health_check_model_groups", lambda r: r._core_background_health_check_model_groups
    )
    enable_tag_filtering = _setting("enable_tag_filtering", lambda r: r._core_enable_tag_filtering)
    tag_filtering_match_any = _setting("tag_filtering_match_any", lambda r: r._core_tag_filtering_match_any)
    tag_routing_prefix = _setting("tag_routing_prefix", lambda r: r._core_tag_routing_prefix)
    router_general_settings = _setting("router_general_settings", lambda r: r._core_router_general_settings)
    provider_budget_config = _setting("provider_budget_config", lambda r: r._core_provider_budget_config)
    provider_default_deployment_ids = _setting(
        "provider_default_deployment_ids", lambda r: r._core_provider_default_deployment_ids
    )
    fallback_access_check = _setting("fallback_access_check", lambda r: r._core_fallback_access_check)
    fallback_budget_check = _setting("fallback_budget_check", lambda r: r._core_fallback_budget_check)
    # These three are declared with the attribute type Router gave them rather than the descriptor's:
    # protocols in router_utils and the proxy match Router on them, and protocol matching does not
    # see through a descriptor. Reads and writes still go through _setting at runtime.
    fallbacks: list | list[dict[str, list[str] | list | None]] | None = _setting(  # pyright: ignore[reportAssignmentType]  # see above  # mutable-ok: the old type
        "fallbacks", lambda r: r._core_fallbacks
    )
    kubernetes_pod_discovery: "KubernetesPodDiscovery" = _setting(  # pyright: ignore[reportAssignmentType]  # see above
        "kubernetes_pod_discovery", lambda r: r._core_kubernetes_pod_discovery
    )
    search_tools: "list[SearchToolTypedDict]" = _setting(  # pyright: ignore[reportAssignmentType]  # see above  # mutable-ok: the old type
        "search_tools", lambda r: r._core_search_tools
    )
    guardrail_list = _setting("guardrail_list", lambda r: r._core_guardrail_list)
    cache_responses = _setting("cache_responses", lambda r: r._core_cache_responses)

    # -- registry: one deployment ---------------------------------------------------------------
    def get_deployment(self, model_id: str) -> "Deployment | None":
        _note("get_deployment", "registry().deployment(id)")
        return _as_router(self)._core_get_deployment(model_id=model_id)

    def has_model_id(self, candidate_id: str) -> bool:
        _note("has_model_id", "registry().has_deployment(id)")
        return _as_router(self)._core_has_model_id(candidate_id)

    def get_model_info(self, id: str) -> dict | None:  # mutable-ok: the old return type, unchanged
        _note("get_model_info", "registry().deployment_dict(id)")
        return _as_router(self)._core_get_model_info(id=id)

    def get_deployment_model_info(self, model_id: str, model_name: str) -> "ModelMapInfo | None":
        _note("get_deployment_model_info", "registry().deployment_model_info(id, model_name)")
        return _as_router(self)._core_get_deployment_model_info(model_id=model_id, model_name=model_name)

    @overload
    def get_router_model_info(
        self,
        deployment: "dict | Deployment",  # mutable-ok: the old parameter type, unchanged
        received_model_name: str,
        id: None = None,
    ) -> "ModelMapInfo": ...

    @overload
    def get_router_model_info(self, deployment: None, received_model_name: str, id: str) -> "ModelMapInfo": ...

    def get_router_model_info(
        self,
        deployment: "dict | Deployment | None",  # mutable-ok: the old parameter type, unchanged
        received_model_name: str,
        id: str | None = None,
    ) -> "ModelMapInfo":
        _note("get_router_model_info", "registry().router_model_info(...)")
        router: Final = _as_router(self)
        # The core's overloads split (deployment, id) by which one is None; this forwards either form unchanged.
        return router._core_get_router_model_info(  # pyright: ignore[reportCallIssue]  # see above
            deployment=deployment,  # pyright: ignore[reportArgumentType]  # see above
            received_model_name=received_model_name,
            id=id,  # pyright: ignore[reportArgumentType]  # see above
        )

    def get_discovered_model_info(self, model_id: str | None) -> Mapping[str, int]:
        _note("get_discovered_model_info", "registry().discovered_model_info(id)")
        return _as_router(self)._core_get_discovered_model_info(model_id)

    # -- registry: many deployments -------------------------------------------------------------
    def get_model_list(
        self, model_name: str | None = None, team_id: str | None = None
    ) -> "list[DeploymentTypedDict] | None":  # mutable-ok: the old return type, unchanged
        _note("get_model_list", "registry().deployments(model_name, team_id=...)")
        return _as_router(self)._core_get_model_list(model_name=model_name, team_id=team_id)

    def get_all_deployments(
        self, model_name: str, model_alias: str | None = None, team_id: str | None = None
    ) -> "list[DeploymentTypedDict]":  # mutable-ok: the old return type, unchanged
        _note("get_all_deployments", "registry().deployments(model_name, source='all')")
        return _as_router(self)._core_get_all_deployments(
            model_name=model_name, model_alias=model_alias, team_id=team_id
        )

    _get_all_deployments = _private_alias("get_all_deployments", get_all_deployments)

    def config_deployments(self) -> Iterator[Mapping[str, object]]:
        _note("config_deployments", "registry().deployments(source='config')")
        return _as_router(self)._core_config_deployments()

    def deployments_named(self, model_name: str) -> "tuple[DeploymentTypedDict, ...] | None":
        _note("deployments_named", "registry().deployments(model_name, source='named')")
        return _as_router(self)._core_deployments_named(model_name)

    def get_model_list_from_model_alias(
        self, model_name: str | None = None
    ) -> "list[DeploymentTypedDict]":  # mutable-ok: the old return type, unchanged
        _note("get_model_list_from_model_alias", "registry().deployments(model_name, source='alias')")
        return _as_router(self)._core_get_model_list_from_model_alias(model_name=model_name)

    def get_model_list_from_routing_groups(self, model_name: str | None = None) -> "Sequence[DeploymentTypedDict]":
        _note("get_model_list_from_routing_groups", "registry().deployments(model_name, source='routing_groups')")
        return _as_router(self)._core_get_model_list_from_routing_groups(model_name=model_name)

    def get_model_list_of_routed_group(
        self, model_group: str
    ) -> "list[DeploymentTypedDict]":  # mutable-ok: the old return type, unchanged
        _note("get_model_list_of_routed_group", "registry().deployments(model_group, source='routed_group')")
        return _as_router(self)._core_get_model_list_of_routed_group(model_group)

    def wildcard_route_deployments(self, model: str, team_id: str | None) -> Sequence[Mapping[str, object]]:
        _note("wildcard_route_deployments", "registry().deployments(model, source='wildcard')")
        return _as_router(self)._core_wildcard_route_deployments(model, team_id)

    def get_routable_deployments(
        self,
        model: str,
        request_kwargs: Mapping[str, object] = MappingProxyType({}),
        call_type: str | None = None,
    ) -> "tuple[DeploymentTypedDict, ...]":
        _note(
            "get_routable_deployments", "registry().deployments(model, source='routable', request=..., call_type=...)"
        )
        return _as_router(self)._core_get_routable_deployments(model, request_kwargs, call_type)

    def get_deployment_by_model_group_name(self, model_group_name: str) -> "Deployment | None":
        _note("get_deployment_by_model_group_name", "registry().deployments(name, source='first_in_group')")
        return _as_router(self)._core_get_deployment_by_model_group_name(model_group_name=model_group_name)

    def deployments_for_request(
        self, model: str, request_kwargs: Mapping[str, object]
    ) -> "Sequence[DeploymentTypedDict]":
        _note("deployments_for_request", "registry().deployments(model, source='for_request', request=...)")
        return _as_router(self)._core_deployments_for_request(model, request_kwargs)

    def should_include_deployment(
        self,
        model_name: str,
        model: "DeploymentTypedDict | dict",  # mutable-ok: the old parameter type, unchanged
        team_id: str | None = None,
    ) -> bool:
        _note("should_include_deployment", "registry().includes(deployment, model_name=..., team_id=...)")
        return _as_router(self)._core_should_include_deployment(model_name=model_name, model=model, team_id=team_id)

    @staticmethod
    def deployment_usable_by_team(model: "Mapping[str, object] | Deployment", team_id: str | None) -> bool:
        _note("deployment_usable_by_team", "registry().includes(deployment, team_id=...)")
        from litellm.router import Router

        return Router._core_deployment_usable_by_team(model, team_id)

    _deployment_usable_by_team = deployment_usable_by_team

    # -- registry: names -------------------------------------------------------------------------
    def get_model_names(self, team_id: str | None = None) -> list[str]:  # mutable-ok: the old return type, unchanged
        _note("get_model_names", "registry().names(team_id=...)")
        return _as_router(self)._core_get_model_names(team_id=team_id)

    def get_model_ids(
        self, model_name: str | None = None, exclude_team_models: bool = False
    ) -> list[str]:  # mutable-ok: the old return type, unchanged
        _note("get_model_ids", "registry().names(model_name, kind='ids')")
        return _as_router(self)._core_get_model_ids(model_name=model_name, exclude_team_models=exclude_team_models)

    def resolved_litellm_models(
        self, model_name: str, team_id: str | None = None, *, use_base_model: bool = False
    ) -> tuple[str, ...]:
        _note("resolved_litellm_models", "registry().names(model_name, kind='litellm_models' or 'litellm_base_models')")
        return _as_router(self)._core_resolved_litellm_models(
            model_name, team_id=team_id, use_base_model=use_base_model
        )

    def wildcard_routed_models(self, model: str, team_id: str | None) -> frozenset[str]:
        _note("wildcard_routed_models", "registry().names(model, kind='wildcard')")
        return _as_router(self)._core_wildcard_routed_models(model, team_id)

    def get_candidate_model_ids_for_route(self, model: str, team_id: str | None = None) -> frozenset[str]:
        _note("get_candidate_model_ids_for_route", "registry().names(model, kind='route_candidates')")
        return _as_router(self)._core_get_candidate_model_ids_for_route(model, team_id=team_id)

    # -- registry: resolving names ---------------------------------------------------------------
    def routable_model_group(self, model_name: str) -> str:
        _note("routable_model_group", "registry().resolve(name)")
        return _as_router(self)._core_routable_model_group(model_name)

    def get_model_from_alias(self, model: str) -> str | None:
        _note("get_model_from_alias", "registry().resolve(name, kind='alias')")
        return _as_router(self)._core_get_model_from_alias(model=model)

    _get_model_from_alias = _private_alias("get_model_from_alias", get_model_from_alias)

    def resolve_served_model_name(self, model_name: str, served_model: str) -> str:
        _note("resolve_served_model_name", "registry().resolve(name, kind='served', served_model=...)")
        return _as_router(self)._core_resolve_served_model_name(model_name, served_model)

    def resolve_model_name_from_model_id(
        self, model_id: str | None, custom_llm_provider: str | None = None
    ) -> str | None:
        _note("resolve_model_name_from_model_id", "registry().resolve(id, kind='from_id')")
        return _as_router(self)._core_resolve_model_name_from_model_id(
            model_id, custom_llm_provider=custom_llm_provider
        )

    def map_team_model(
        self, team_model_name: str | None, team_id: str, custom_llm_provider: str | None = None
    ) -> str | None:
        _note("map_team_model", "registry().resolve(name, kind='team_model', team_id=...)")
        return _as_router(self)._core_map_team_model(team_model_name, team_id, custom_llm_provider=custom_llm_provider)

    def is_recognized_model(self, model: str) -> bool:
        _note("is_recognized_model", "registry().resolve(name, kind='recognized')")
        return _as_router(self)._core_is_recognized_model(model)

    def get_routable_upstream_model(self, model_name: str, team_id: str | None = None) -> str | None:
        _note("get_routable_upstream_model", "registry().resolve(name, kind='upstream')")
        return _as_router(self)._core_get_routable_upstream_model(model_name, team_id=team_id)

    def get_search_tool_support(self, model_name: str, team_id: str | None = None) -> tuple[bool | None, ...]:
        _note("get_search_tool_support", "registry().resolve(name, kind='search_tool_support')")
        return _as_router(self)._core_get_search_tool_support(model_name, team_id=team_id)

    def get_model_group(self, id: str) -> list | None:  # mutable-ok: the old return type, unchanged
        _note("get_model_group", "registry().resolve(id, kind='group_of_id')")
        return _as_router(self)._core_get_model_group(id=id)

    def is_model_access_group_for_wildcard_route(self, model_access_group: str) -> bool:
        _note("is_model_access_group_for_wildcard_route", "registry().resolve(group, kind='wildcard_access_group')")
        return _as_router(self)._core_is_model_access_group_for_wildcard_route(model_access_group=model_access_group)

    _is_model_access_group_for_wildcard_route = _private_alias(
        "is_model_access_group_for_wildcard_route", is_model_access_group_for_wildcard_route
    )

    def team_model_has_alternatives(self, deployment_id: str) -> bool:
        _note("team_model_has_alternatives", "registry().resolve(id, kind='team_model_alternatives')")
        return _as_router(self)._core_team_model_has_alternatives(deployment_id)

    def routing_group_has_alternatives(self, model_group: str | None) -> bool:
        _note("routing_group_has_alternatives", "registry().resolve(group, kind='routing_group_alternatives')")
        return _as_router(self)._core_routing_group_has_alternatives(model_group)

    # -- registry: groups, access groups, credentials, plugins -----------------------------------
    def get_model_group_info(
        self, model_group: str, *, include_hidden: bool = False, team_id: str | None = None
    ) -> "ModelGroupInfo | None":
        _note("get_model_group_info", "registry().group(name)")
        return _as_router(self)._core_get_model_group_info(model_group, include_hidden=include_hidden, team_id=team_id)

    def cached_model_group_info(self, model_group: str) -> "ModelGroupInfo | None":
        _note("cached_model_group_info", "registry().group(name, field='cached_info')")
        return _as_router(self)._core_cached_model_group_info(model_group)

    def get_configured_display_name(self, model_name: str) -> str | None:
        _note("get_configured_display_name", "registry().group(name, field='display_name')")
        return _as_router(self)._core_get_configured_display_name(model_name)

    def get_configured_mode(self, model_name: str) -> str | None:
        _note("get_configured_mode", "registry().group(name, field='mode')")
        return _as_router(self)._core_get_configured_mode(model_name)

    def get_configured_service_tiers(self, model_name: str, team_id: str | None = None) -> tuple[object, ...]:
        _note("get_configured_service_tiers", "registry().group(name, field='service_tiers')")
        return _as_router(self)._core_get_configured_service_tiers(model_name, team_id=team_id)

    def get_configured_token_limits(self, model_name: str) -> tuple[int | None, int | None]:
        _note("get_configured_token_limits", "registry().group(name, field='token_limits')")
        return _as_router(self)._core_get_configured_token_limits(model_name)

    def get_model_listing_info(self, model_name: str) -> "DeploymentModelListingInfo | None":
        _note("get_model_listing_info", "registry().group(name, field='listing_info')")
        return _as_router(self)._core_get_model_listing_info(model_name)

    def get_model_access_groups(
        self,
        model_name: str | None = None,
        model_access_group: str | None = None,
        team_id: str | None = None,
    ) -> dict[str, list[str]]:  # mutable-ok: the old return type, unchanged
        _note("get_model_access_groups", "registry().access_groups(...)")
        return _as_router(self)._core_get_model_access_groups(
            model_name=model_name, model_access_group=model_access_group, team_id=team_id
        )

    def get_deployment_credentials(self, model_id: str) -> dict | None:  # mutable-ok: the old return type, unchanged
        _note("get_deployment_credentials", "registry().credentials(id)")
        return _as_router(self)._core_get_deployment_credentials(model_id=model_id)

    def get_deployment_credentials_with_provider(
        self, model_id: str, team_id: str | None = None
    ) -> "dict[str, Any] | None":  # pyright: ignore[reportExplicitAny]  # the old return type, unchanged  # mutable-ok: same
        _note("get_deployment_credentials_with_provider", "registry().credentials(id, form='with_provider')")
        return _as_router(self)._core_get_deployment_credentials_with_provider(model_id=model_id, team_id=team_id)

    def get_credential_deployment(self, model_id: str, team_id: str | None = None) -> "Deployment | None":
        _note("get_credential_deployment", "registry().credentials(id, form='deployment')")
        return _as_router(self)._core_get_credential_deployment(model_id=model_id, team_id=team_id)

    def get_router_deployment_budget_limiter(self) -> "RouterBudgetLimiting | None":
        _note("get_router_deployment_budget_limiter", "registry().plugins('budget_limiter')")
        return _as_router(self)._core_get_router_deployment_budget_limiter()

    _get_router_deployment_budget_limiter = _private_alias(
        "get_router_deployment_budget_limiter", get_router_deployment_budget_limiter
    )

    @staticmethod
    def generate_model_id(model_group: str, litellm_params: dict) -> str:  # mutable-ok: the old parameter type
        _note("generate_model_id", "litellm.router_api.deployment_id(model_group, litellm_params)")
        from litellm.router_api import deployment_id

        return deployment_id(model_group, litellm_params)

    # -- settings: methods -----------------------------------------------------------------------
    def get_settings(self) -> dict[str, object]:  # mutable-ok: Router._core_get_settings's type, a fresh dict
        _note("get_settings", "settings().as_dict()")
        return _as_router(self)._core_get_settings()

    def get_allowed_fails_from_policy(self, exception: Exception) -> int | None:
        _note("get_allowed_fails_from_policy", "settings().allowed_fails_for(exception)")
        return _as_router(self)._core_get_allowed_fails_from_policy(exception=exception)

    def get_available_guardrail(self, guardrail_name: str) -> "GuardrailTypedDict":
        _note("get_available_guardrail", "settings().guardrail(name)")
        return _as_router(self)._core_get_available_guardrail(guardrail_name)

    # -- health ----------------------------------------------------------------------------------
    @staticmethod
    def is_deployment_blocked(deployment: "Deployment") -> bool:
        _note("is_deployment_blocked", "health().is_blocked(deployment)")
        from litellm.router_api import RouterHealth

        return RouterHealth.is_blocked(deployment)

    _is_deployment_blocked = is_deployment_blocked

    @staticmethod
    def are_all_deployments_blocked(deployments: "Sequence[DeploymentTypedDict]") -> bool:
        _note("are_all_deployments_blocked", "health().all_blocked(deployments)")
        from litellm.router_api import RouterHealth

        return RouterHealth.all_blocked(deployments)

    _are_all_deployments_blocked = are_all_deployments_blocked

    def get_fully_blocked_model_names(self) -> set[str]:  # mutable-ok: the old return type, unchanged
        _note("get_fully_blocked_model_names", "health().fully_blocked_model_names()")
        return _as_router(self)._core_get_fully_blocked_model_names()

    async def async_get_fully_unhealthy_model_names(self) -> set[str]:  # mutable-ok: the old return type, unchanged
        _note("async_get_fully_unhealthy_model_names", "health().fully_unhealthy_model_names()")
        return await _as_router(self)._core_async_get_fully_unhealthy_model_names()

    async def get_model_group_usage(self, model_group: str) -> tuple[int | None, int | None]:
        _note("get_model_group_usage", "health().usage(group)")
        return await _as_router(self)._core_get_model_group_usage(model_group=model_group)

    async def get_remaining_model_group_usage(
        self, model_group: str
    ) -> dict[str, int]:  # mutable-ok: the old return type, unchanged
        _note("get_remaining_model_group_usage", "health().usage(group, kind='remaining')")
        return await _as_router(self)._core_get_remaining_model_group_usage(model_group=model_group)

    def team_ptu_ceiling(self, team_id: str, model_group: str) -> "PTUTeamCeiling | None":
        _note("team_ptu_ceiling", "health().team_ptu_ceiling(team_id, group)")
        return _as_router(self)._core_team_ptu_ceiling(team_id, model_group)

    def split_request_priority(
        self, model: str, kwargs: Mapping[str, object]
    ) -> tuple[int | None, Mapping[str, object]]:
        _note("split_request_priority", "health().split_priority(model, kwargs)")
        return _as_router(self)._core_split_request_priority(model, kwargs)

    # -- selection -------------------------------------------------------------------------------
    async def async_get_healthy_deployments(
        self,
        model: str,
        request_kwargs: dict,  # mutable-ok: the old parameter type, unchanged
        messages: list[dict[str, str]] | None = None,  # mutable-ok: the old parameter type, unchanged
        input: str | list | None = None,  # mutable-ok: the old parameter type, unchanged
        specific_deployment: bool | None = False,
        parent_otel_span: "Span | None" = None,
        health_check_probe: bool = False,
    ) -> list[dict] | dict:  # mutable-ok: the old return type, unchanged
        _note("async_get_healthy_deployments", "select(model, request_kwargs, kind='healthy')")
        return await _as_router(self)._core_async_get_healthy_deployments(
            model=model,
            request_kwargs=request_kwargs,
            messages=messages,
            input=input,
            specific_deployment=specific_deployment,
            parent_otel_span=parent_otel_span,
            health_check_probe=health_check_probe,
        )

    def get_available_deployment_for_pass_through(  # noqa: ANN201  # the old member had no return annotation; its inferred type is kept by inferring it from the core
        self,
        model: str,
        messages: list[dict[str, str]] | None = None,  # mutable-ok: the old parameter type, unchanged
        input: str | list | None = None,  # mutable-ok: the old parameter type, unchanged
        specific_deployment: bool | None = False,
        request_kwargs: dict | None = None,  # mutable-ok: the old parameter type, unchanged
    ):
        _note("get_available_deployment_for_pass_through", "select(model, kind='pass_through')")
        return _as_router(self)._core_get_available_deployment_for_pass_through(
            model=model,
            messages=messages,
            input=input,
            specific_deployment=specific_deployment,
            request_kwargs=request_kwargs,
        )

    # -- writes ----------------------------------------------------------------------------------
    def upsert_deployment(self, deployment: "Deployment") -> "Deployment | None":
        _note("upsert_deployment", "apply(ChangeSet(upsert=[deployment]))")
        return _as_router(self).apply(ChangeSet(upsert=[deployment])).upserted[0]

    def delete_deployment(self, id: str) -> "Deployment | None":
        _note("delete_deployment", "apply(ChangeSet(delete=[id]))")
        return _as_router(self).apply(ChangeSet(delete=[id])).deleted[0]

    def update_settings(self, **kwargs: object) -> None:  # kwargs-ok: the old signature, unchanged
        _note("update_settings", "apply(ChangeSet(settings={...}))")
        _as_router(self).apply(ChangeSet(settings=kwargs))

    def update_redis_cache(self, cache: "RedisCache") -> None:
        _note("update_redis_cache", "apply(ChangeSet(redis_cache=cache))")
        _as_router(self).apply(ChangeSet(redis_cache=cache))

    _update_redis_cache = _private_alias("update_redis_cache", update_redis_cache)

    async def arefresh_model_info(self, *, client: "AsyncHTTPHandler | None" = None) -> None:
        _note("arefresh_model_info", "aapply(ChangeSet(refresh_model_info=True))")
        await _as_router(self).aapply(ChangeSet(refresh_model_info=True, refresh_client=client))

    # -- plugins, observers, lifecycle -----------------------------------------------------------
    def set_custom_routing_strategy(self, CustomRoutingStrategy: "CustomRoutingStrategyBase") -> None:
        _note("set_custom_routing_strategy", "register_plugin(strategy)")
        _as_router(self).register_plugin(CustomRoutingStrategy)

    def serve_through_request_logging(self) -> None:
        _note("serve_through_request_logging", "observe(request_scoped=True)")
        _as_router(self).observe(request_scoped=True)

    def bind_to_request_logging(self, logging_obj: "LiteLLMLogging") -> None:
        _note("bind_to_request_logging", "observe(logging_obj)")
        # Straight to the core, as before: observe() skips binding when logging_obj is None, the old member did not.
        _as_router(self)._core_bind_to_request_logging(logging_obj)

    def discard(self) -> None:
        _note("discard", "retire()")
        _as_router(self).retire()

    # -- plugin context (router strategies) ------------------------------------------------------
    def arm_routing_read_prefetch(
        self,
        model: str,
        request_kwargs: dict[str, object] | None = None,  # mutable-ok: the old parameter type, unchanged
    ) -> None:
        _note("arm_routing_read_prefetch", "RoutingContext.arm_read_prefetch")
        _as_router(self)._core_arm_routing_read_prefetch(model, request_kwargs)

    async def async_pre_routing_hook(
        self,
        model: str,
        request_kwargs: dict[str, object],  # mutable-ok: the old parameter type, unchanged
        messages: "list[dict[str, Any]] | None" = None,  # pyright: ignore[reportExplicitAny]  # the old type  # mutable-ok: same
        input: str | list | None = None,  # mutable-ok: the old parameter type, unchanged
        specific_deployment: bool | None = False,
    ) -> "PreRoutingHookResponse | None":
        _note("async_pre_routing_hook", "RoutingContext.pre_routing_hook")
        return await _as_router(self)._core_async_pre_routing_hook(
            model=model,
            request_kwargs=request_kwargs,
            messages=messages,
            input=input,
            specific_deployment=specific_deployment,
        )

    def log_retry(self, kwargs: dict, e: Exception) -> dict:  # mutable-ok: the old signature, unchanged
        _note("log_retry", "RoutingContext.log_retry")
        return _as_router(self)._core_log_retry(kwargs=kwargs, e=e)


def _name_members_after_router() -> None:
    # The old members were Router methods, so errors that name the method (a wrong-arity TypeError,
    # or "litellm.router.Router.<name>() argument after ** must be a mapping") keep naming it the
    # way they did before the members moved here.
    for member in vars(DeprecatedRouterMembers).values():
        function = member.__func__ if isinstance(member, staticmethod) else member
        if inspect.isfunction(function):
            function.__qualname__ = f"Router.{function.__name__}"
            function.__module__ = "litellm.router"


_name_members_after_router()
