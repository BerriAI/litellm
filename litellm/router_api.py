"""
The Router's entry points and the typed views they return.

``litellm.Router`` exposes its endpoint methods (``acompletion``, ``aembedding``, ...) plus eight
entry points, and every other read or write that code outside the router needs goes through one
of them:

- ``registry()`` -> ``RouterRegistry``: the configured deployments, indexed by id, model group,
  alias, wildcard pattern and team, and the plugins they configure.
- ``settings()`` -> ``RouterSettings``: one frozen snapshot of the configuration a caller may read.
- ``health()`` -> ``RouterHealth``: cooldowns, blocked deployments, health checks and usage.
- ``select(model, ...)``: pick a deployment without calling it.
- ``apply(ChangeSet)`` / ``aapply(ChangeSet)``: runtime changes (deployments, settings, plugins),
  returning an ``ApplyResult`` with the new registry version.
- ``observe(...)``: bind the router's routing observers to request logging.
- ``register_plugin(plugin)``: a custom routing strategy.
- ``retire()``: release the router's global registrations.

The views read the router's state when they are called; they hold no copy of it. A Rust core
implements the same views (see litellm-rust/crates/router-core). Each query names the old Router
member it replaces in its docstring; ``DeprecatedRouterMembers`` keeps the old names working, typed
as they were.
"""

# pyright: reportPrivateUsage=false, reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportMissingTypeArgument=false, reportUnknownParameterType=false
# The views are part of the router: they read and write the Router's ``_core_*`` members by design
# (see the router-core boundary in litellm/router.py), the same way Router's own methods do. Each
# selector query (``deployments(source=...)``, ``names(kind=...)``, ...) is overloaded per selector
# value and returns exactly the type of the ``_core_*`` member that value reads, which is the type
# the old public member had, so moving a caller onto the entry point keeps its types. Some of those
# members are still untyped (bare ``dict`` / ``list``) or mutable (the live objects); the overloads
# repeat them unchanged, and typing the members themselves is the core's job.

import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, TypeAlias, overload

if TYPE_CHECKING:
    from typing import Any  # noqa: TID251  # only to spell the cores' untyped (Unknown) containers and one Any type

    from litellm.caching.caching import DualCache, RedisCache
    from litellm.integrations.SlackAlerting.slack_alerting import SlackAlerting
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
    from litellm.router import Router
    from litellm.router_strategy.adaptive_router.adaptive_router import AdaptiveRouter
    from litellm.router_strategy.auto_router.auto_router import AutoRouter
    from litellm.router_strategy.budget_limiter import RouterBudgetLimiting
    from litellm.router_strategy.complexity_router.complexity_router import ComplexityRouter
    from litellm.router_strategy.quality_router.quality_router import QualityRouter
    from litellm.router_utils.cooldown_cache import CooldownCache
    from litellm.router_utils.health_state_cache import DeploymentHealthCache
    from litellm.router_utils.kubernetes_pod_discovery import KubernetesPodDiscovery
    from litellm.router_utils.pattern_match_deployments import PatternMatchRouter
    from litellm.router_utils.ptu_shares import PTUTeamCeiling
    from litellm.types.router import (
        AllowedFailsPolicy,
        Deployment,
        DeploymentModelListingInfo,
        DeploymentTypedDict,
        FallbackAccessCheck,
        FallbackBudgetCheck,
        GuardrailTypedDict,
        ModelGroupInfo,
        PreRoutingHookResponse,
        RouterGeneralSettings,
        RouterModelGroupAliasItem,
        RoutingGroup,
        RoutingPlugin,
        SearchToolTypedDict,
        TaggedPreRoutingStrategy,
    )
    from litellm.types.utils import BudgetConfig
    from litellm.types.utils import ModelInfo as ModelMapInfo

    # The cores' untyped containers (bare ``list`` / ``dict``, element types Unknown), spelled with Any:
    # an overload set with one partially-unknown overload makes every call through it partially unknown.
    UntypedList: TypeAlias = list[Any]  # pyright: ignore[reportExplicitAny]  # see above
    UntypedDict: TypeAlias = dict[Any, Any]  # pyright: ignore[reportExplicitAny]  # see above

    # What registry().plugins(kind) returns, one member per kind.
    RegistryPlugins: TypeAlias = (
        dict[str, list[TaggedPreRoutingStrategy[AutoRouter]]]
        | dict[str, list[TaggedPreRoutingStrategy[ComplexityRouter]]]
        | dict[str, list[TaggedPreRoutingStrategy[QualityRouter]]]
        | dict[str, list[TaggedPreRoutingStrategy[AdaptiveRouter]]]
        | list[RoutingPlugin]
        | RouterBudgetLimiting
        | DualCache
        | None
    )


DeploymentSource: TypeAlias = Literal[
    "group",
    "all",
    "config",
    "named",
    "alias",
    "routing_groups",
    "routed_group",
    "wildcard",
    "routable",
    "first_in_group",
    "for_request",
]
NameKind: TypeAlias = Literal[
    "ids",
    "groups",
    "group_set",
    "deployment_names",
    "team_public",
    "litellm_models",
    "litellm_base_models",
    "wildcard",
    "route_candidates",
]
ResolveKind: TypeAlias = Literal[
    "group",
    "alias",
    "served",
    "from_id",
    "team_model",
    "recognized",
    "upstream",
    "group_of_id",
    "wildcard_access_group",
    "team_model_alternatives",
    "routing_group_alternatives",
    "search_tool_support",
]
GroupField: TypeAlias = Literal[
    "info",
    "cached_info",
    "display_name",
    "mode",
    "service_tiers",
    "token_limits",
    "listing_info",
]
CredentialForm: TypeAlias = Literal["params", "with_provider", "deployment"]
IndexName: TypeAlias = Literal["model_name", "team_model"]
PluginKind: TypeAlias = Literal[
    "auto",
    "complexity",
    "quality",
    "adaptive",
    "routing",
    "budget_limiter",
    "prompt_caching_pins",
]
SelectKind: TypeAlias = Literal["pick", "healthy", "pass_through"]
UsageKind: TypeAlias = Literal["current", "remaining"]


def deployment_usable_by_team(deployment: "Mapping[str, object] | Deployment", team_id: str | None) -> bool:
    """
    Whether ``team_id`` may use ``deployment``: a team-scoped deployment (``model_info.team_id``
    set) only by that team, any other deployment by everyone. Needs no router instance.
    """
    from litellm.router import Router

    return Router._core_deployment_usable_by_team(deployment, team_id)


def deployment_id(model_group: str, litellm_params: Mapping[str, object]) -> str:
    """
    The stable id of a deployment that has no ``model_info.id``: a sha256 over the model group and
    every litellm_params key and value, in order. Replaces ``Router.generate_model_id``.
    """
    parts: Final = [model_group]
    for k, v in litellm_params.items():
        for item in (k, v):
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(json.dumps(item, default=_json_default_stable_id))
            else:
                parts.append(str(item))
    return hashlib.sha256("".join(parts).encode()).hexdigest()


def _json_default_stable_id(obj: object) -> str:
    from litellm.router import Router

    return Router._json_default_stable_id(obj)


class RouterRegistry:
    """
    Read-only queries over the configured deployments (``registry()``).

    ``version`` changes whenever the deployments change, so a caller can tell whether
    what it read earlier is still current.
    """

    __slots__ = ("_router",)

    def __init__(self, router: "Router") -> None:
        self._router = router

    @property
    def version(self) -> int:
        return self._router._registry_version

    # -- one deployment -------------------------------------------------------------------------

    def deployment(self, id: str) -> "Deployment | None":
        """The deployment with this ``model_info.id`` (was ``get_deployment``)."""
        return self._router._core_get_deployment(model_id=id)

    def has_deployment(self, id: str) -> bool:
        """Whether a deployment has this id (was ``has_model_id``)."""
        return self._router._core_has_model_id(id)

    def deployment_dict(
        self, id: str
    ) -> "UntypedDict | None":  # a mutable list/dict, as before: Router._core_get_model_info's type, the live dict
        """The deployment as its model_list dict (was ``get_model_info``)."""
        return self._router._core_get_model_info(id=id)

    def deployment_model_info(self, id: str, model_name: str, *, cached: bool = False) -> "ModelMapInfo | None":
        """
        Model info for a deployment, merged with the pricing map (was
        ``get_deployment_model_info``; ``cached=True`` was ``cached_deployment_model_info``).
        """
        if cached:
            return self._router._core_cached_deployment_model_info(id, model_name)
        return self._router._core_get_deployment_model_info(model_id=id, model_name=model_name)

    def router_model_info(
        self,
        deployment: "dict[str, object] | Deployment | None",  # mutable-ok: the core takes a dict
        received_model_name: str,
        id: str | None = None,
    ) -> "ModelMapInfo":
        """Model info used for routing decisions (was ``get_router_model_info``)."""
        r: Final = self._router
        # The core's overloads split (deployment, id) by which one is None; this forwards either form unchanged.
        return r._core_get_router_model_info(  # pyright: ignore[reportCallIssue]  # see above
            deployment=deployment,  # pyright: ignore[reportArgumentType]  # see above
            received_model_name=received_model_name,
            id=id,  # pyright: ignore[reportArgumentType]  # see above
        )

    def discovered_model_info(self, id: str | None) -> Mapping[str, int]:
        """Limits discovered from the provider for a deployment (was ``get_discovered_model_info``)."""
        return self._router._core_get_discovered_model_info(id)

    def zero_cost(self) -> Mapping[str, bool] | None:
        """The per-deployment zero-cost lookup the auth checks read (was ``_zero_cost_cache``)."""
        return getattr(self._router, "_zero_cost_cache", None)

    # -- many deployments -----------------------------------------------------------------------

    @overload
    def deployments(
        self,
        model_name: str | None = None,
        *,
        source: Literal["group"] = "group",
        team_id: str | None = None,
        model_alias: str | None = None,
        request: Mapping[str, object] | None = None,
        call_type: str | None = None,
    ) -> "list[DeploymentTypedDict] | None": ...  # mutable-ok: Router._core_get_model_list's type, the live lists

    @overload
    def deployments(
        self,
        model_name: str | None = None,
        *,
        source: Literal["alias"],
        team_id: str | None = None,
        model_alias: str | None = None,
        request: Mapping[str, object] | None = None,
        call_type: str | None = None,
    ) -> "list[DeploymentTypedDict]": ...  # mutable-ok: Router._core_get_model_list_from_model_alias's type

    @overload
    def deployments(
        self,
        model_name: str,
        *,
        source: Literal["all", "routed_group"],
        team_id: str | None = None,
        model_alias: str | None = None,
        request: Mapping[str, object] | None = None,
        call_type: str | None = None,
    ) -> "list[DeploymentTypedDict]": ...  # mutable-ok: the cores' type (get_all_deployments, ...)

    @overload
    def deployments(
        self,
        model_name: str | None = None,
        *,
        source: Literal["config"],
        team_id: str | None = None,
        model_alias: str | None = None,
        request: Mapping[str, object] | None = None,
        call_type: str | None = None,
    ) -> Iterator[Mapping[str, object]]: ...

    @overload
    def deployments(
        self,
        model_name: str | None = None,
        *,
        source: Literal["routing_groups"],
        team_id: str | None = None,
        model_alias: str | None = None,
        request: Mapping[str, object] | None = None,
        call_type: str | None = None,
    ) -> "Sequence[DeploymentTypedDict]": ...

    @overload
    def deployments(
        self,
        model_name: str,
        *,
        source: Literal["for_request"],
        team_id: str | None = None,
        model_alias: str | None = None,
        request: Mapping[str, object] | None = None,
        call_type: str | None = None,
    ) -> "Sequence[DeploymentTypedDict]": ...

    @overload
    def deployments(
        self,
        model_name: str,
        *,
        source: Literal["named"],
        team_id: str | None = None,
        model_alias: str | None = None,
        request: Mapping[str, object] | None = None,
        call_type: str | None = None,
    ) -> "tuple[DeploymentTypedDict, ...] | None": ...

    @overload
    def deployments(
        self,
        model_name: str,
        *,
        source: Literal["routable"],
        team_id: str | None = None,
        model_alias: str | None = None,
        request: Mapping[str, object] | None = None,
        call_type: str | None = None,
    ) -> "tuple[DeploymentTypedDict, ...]": ...

    @overload
    def deployments(
        self,
        model_name: str,
        *,
        source: Literal["wildcard"],
        team_id: str | None = None,
        model_alias: str | None = None,
        request: Mapping[str, object] | None = None,
        call_type: str | None = None,
    ) -> Sequence[Mapping[str, object]]: ...

    @overload
    def deployments(
        self,
        model_name: str,
        *,
        source: Literal["first_in_group"],
        team_id: str | None = None,
        model_alias: str | None = None,
        request: Mapping[str, object] | None = None,
        call_type: str | None = None,
    ) -> "Deployment | None": ...

    def deployments(
        self,
        model_name: str | None = None,
        *,
        source: DeploymentSource = "group",
        team_id: str | None = None,
        model_alias: str | None = None,
        request: Mapping[str, object] | None = None,
        call_type: str | None = None,
    ) -> "Sequence[DeploymentTypedDict] | Sequence[Mapping[str, object]] | Iterator[Mapping[str, object]] | Deployment | None":
        """
        Deployments by name. ``source`` picks which index answers:

        - ``group``: the group, its aliases, routing groups, then wildcards; every deployment when
          ``model_name`` is None (was ``get_model_list``)
        - ``all``: every deployment of the group, with ``model_alias`` (was ``get_all_deployments``)
        - ``config``: deployments that came from config rather than the DB (was ``config_deployments``)
        - ``named``: the group index entry, or None (was ``deployments_named``)
        - ``alias`` / ``routing_groups`` / ``routed_group``: (was ``get_model_list_from_model_alias``,
          ``get_model_list_from_routing_groups``, ``get_model_list_of_routed_group``)
        - ``wildcard``: deployments a wildcard route serves (was ``wildcard_route_deployments``)
        - ``routable``: deployments that pass the routing checks for ``request`` and, when given,
          the endpoint ``call_type`` (was ``get_routable_deployments``)
        - ``first_in_group``: the first deployment of the group (was ``get_deployment_by_model_group_name``)
        - ``for_request``: the deployments a request may use (was ``deployments_for_request``)
        """
        r: Final = self._router
        if source == "group":
            return r._core_get_model_list(model_name=model_name, team_id=team_id)
        if source == "config":
            return r._core_config_deployments()
        if source == "alias":
            return r._core_get_model_list_from_model_alias(model_name=model_name)
        if source == "routing_groups":
            return r._core_get_model_list_from_routing_groups(model_name=model_name)
        if model_name is None:
            raise TypeError(f"registry().deployments(source={source!r}) needs a model_name")
        if source == "all":
            return r._core_get_all_deployments(model_name=model_name, model_alias=model_alias, team_id=team_id)
        if source == "named":
            return r._core_deployments_named(model_name)
        if source == "routed_group":
            return r._core_get_model_list_of_routed_group(model_name, team_id)
        if source == "wildcard":
            return r._core_wildcard_route_deployments(model_name, team_id)
        if source == "routable":
            return r._core_get_routable_deployments(
                model_name, MappingProxyType({}) if request is None else request, call_type
            )
        if source == "first_in_group":
            return r._core_get_deployment_by_model_group_name(model_group_name=model_name)
        if source == "for_request":
            return r._core_deployments_for_request(model_name, request or {})
        raise ValueError(f"unknown deployment source {source!r}")

    def all(self) -> "UntypedList":  # a mutable list/dict, as before: the model_list attribute's type, the live list
        """The live model list, in configured order (was the ``model_list`` attribute)."""
        return self._router._core_model_list

    def includes(
        self,
        deployment: "DeploymentTypedDict | dict[str, object]",  # mutable-ok: the core takes a dict
        *,
        model_name: str | None = None,
        team_id: str | None = None,
    ) -> bool:
        """
        Whether a deployment serves ``model_name`` for ``team_id`` (was
        ``should_include_deployment``), or, without ``model_name``, whether the team may use it at
        all (was ``deployment_usable_by_team``).
        """
        from litellm.router import Router

        if model_name is None:
            return Router._core_deployment_usable_by_team(deployment, team_id)
        return self._router._core_should_include_deployment(model_name=model_name, model=deployment, team_id=team_id)

    @overload
    def index(self, name: Literal["model_name"]) -> dict[str, list[int]]: ...  # mutable-ok: the live index

    @overload
    def index(self, name: Literal["team_model"]) -> dict[tuple[str, str], list[int]]: ...  # mutable-ok: the live index

    def index(self, name: IndexName) -> Mapping[str, Sequence[int]] | Mapping[tuple[str, str], Sequence[int]]:
        """
        The raw name indexes: ``model_name`` maps a group name to model_list positions (was
        ``model_name_to_deployment_indices``), ``team_model`` maps (team, public name) keys (was
        ``team_model_to_deployment_indices``).
        """
        if name == "model_name":
            return self._router._core_model_name_to_deployment_indices
        if name == "team_model":
            return self._router._core_team_model_to_deployment_indices
        raise ValueError(f"unknown index {name!r}")

    # -- names ----------------------------------------------------------------------------------

    @overload
    def names(
        self,
        model_name: str | None = None,
        *,
        kind: Literal["groups", "ids"] = "groups",
        team_id: str | None = None,
        exclude_team_models: bool = False,
    ) -> list[str]: ...  # mutable-ok: the cores' type (get_model_names, get_model_ids), a fresh list

    @overload
    def names(
        self,
        model_name: str | None = None,
        *,
        kind: Literal["group_set"],
        team_id: str | None = None,
        exclude_team_models: bool = False,
    ) -> set[str]: ...  # mutable-ok: the model_names attribute's type, the live set

    @overload
    def names(
        self,
        model_name: str | None = None,
        *,
        kind: Literal["deployment_names"],
        team_id: str | None = None,
        exclude_team_models: bool = False,
    ) -> "UntypedList": ...  # a mutable list/dict, as before: the deployment_names attribute's type, the live list

    @overload
    def names(
        self,
        model_name: str | None = None,
        *,
        kind: Literal["team_public"],
        team_id: str | None = None,
        exclude_team_models: bool = False,
    ) -> frozenset[str]: ...

    @overload
    def names(
        self,
        model_name: str,
        *,
        kind: Literal["wildcard", "route_candidates"],
        team_id: str | None = None,
        exclude_team_models: bool = False,
    ) -> frozenset[str]: ...

    @overload
    def names(
        self,
        model_name: str,
        *,
        kind: Literal["litellm_models", "litellm_base_models"],
        team_id: str | None = None,
        exclude_team_models: bool = False,
    ) -> tuple[str, ...]: ...

    def names(
        self,
        model_name: str | None = None,
        *,
        kind: NameKind = "groups",
        team_id: str | None = None,
        exclude_team_models: bool = False,
    ) -> Sequence[str] | AbstractSet[str]:
        """
        Names known to the router. ``kind``:

        - ``groups``: every routable model name, with team public names for ``team_id``
          (was ``get_model_names``)
        - ``group_set``: the set of configured group names (was the ``model_names`` attribute)
        - ``ids``: deployment ids, of one group or all (was ``get_model_ids``)
        - ``deployment_names``: (was the ``deployment_names`` attribute)
        - ``team_public``: team public model names (was ``team_public_model_names``)
        - ``litellm_models``: upstream model names a group resolves to (was ``resolved_litellm_models``)
        - ``litellm_base_models``: the same, with each deployment's ``base_model`` where it sets one
          (was ``resolved_litellm_models(..., use_base_model=True)``)
        - ``wildcard``: names a wildcard route serves (was ``wildcard_routed_models``)
        - ``route_candidates``: deployment ids a request for the name may reach
          (was ``get_candidate_model_ids_for_route``)
        """
        r: Final = self._router
        if kind == "groups":
            return r._core_get_model_names(team_id=team_id)
        if kind == "group_set":
            return r._core_model_names
        if kind == "ids":
            return r._core_get_model_ids(model_name=model_name, exclude_team_models=exclude_team_models)
        if kind == "deployment_names":
            return r._core_deployment_names
        if kind == "team_public":
            return r._core_team_public_model_names
        if model_name is None:
            raise TypeError(f"registry().names(kind={kind!r}) needs a model_name")
        if kind == "litellm_models":
            return r._core_resolved_litellm_models(model_name, team_id=team_id)
        if kind == "litellm_base_models":
            return r._core_resolved_litellm_models(model_name, team_id=team_id, use_base_model=True)
        if kind == "wildcard":
            return r._core_wildcard_routed_models(model_name, team_id)
        if kind == "route_candidates":
            return r._core_get_candidate_model_ids_for_route(model_name, team_id=team_id)
        raise ValueError(f"unknown name kind {kind!r}")

    # -- resolving a name -----------------------------------------------------------------------

    @overload
    def resolve(
        self,
        name: str,
        *,
        kind: Literal["group", "served"] = "group",
        team_id: str | None = None,
        custom_llm_provider: str | None = None,
        served_model: str | None = None,
    ) -> str: ...

    @overload
    def resolve(
        self,
        name: str,
        *,
        kind: Literal["alias", "upstream"],
        team_id: str | None = None,
        custom_llm_provider: str | None = None,
        served_model: str | None = None,
    ) -> str | None: ...

    @overload
    def resolve(
        self,
        name: str,
        *,
        kind: Literal["search_tool_support"],
        team_id: str | None = None,
        custom_llm_provider: str | None = None,
        served_model: str | None = None,
    ) -> tuple[bool | None, ...]: ...

    @overload
    def resolve(
        self,
        name: str | None,
        *,
        kind: Literal["from_id", "team_model"],
        team_id: str | None = None,
        custom_llm_provider: str | None = None,
        served_model: str | None = None,
    ) -> str | None: ...

    @overload
    def resolve(
        self,
        name: str,
        *,
        kind: Literal["recognized", "wildcard_access_group", "team_model_alternatives"],
        team_id: str | None = None,
        custom_llm_provider: str | None = None,
        served_model: str | None = None,
    ) -> bool: ...

    @overload
    def resolve(
        self,
        name: str | None,
        *,
        kind: Literal["routing_group_alternatives"],
        team_id: str | None = None,
        custom_llm_provider: str | None = None,
        served_model: str | None = None,
    ) -> bool: ...

    @overload
    def resolve(
        self,
        name: str,
        *,
        kind: Literal["group_of_id"],
        team_id: str | None = None,
        custom_llm_provider: str | None = None,
        served_model: str | None = None,
    ) -> "UntypedList | None": ...  # a mutable list/dict, as before: Router._core_get_model_group's type, unchanged

    def resolve(
        self,
        name: str | None,
        *,
        kind: ResolveKind = "group",
        team_id: str | None = None,
        custom_llm_provider: str | None = None,
        served_model: str | None = None,
    ) -> str | bool | Sequence[object] | None:
        """
        Resolve a requested name. ``kind``:

        - ``group``: the model group a name routes to (was ``routable_model_group``)
        - ``alias``: the target of a model_group_alias, or None (was ``get_model_from_alias``)
        - ``served``: the name to report for ``served_model`` (was ``resolve_served_model_name``)
        - ``from_id``: the model name of a deployment id (was ``resolve_model_name_from_model_id``)
        - ``team_model``: a team's public name to its group (was ``map_team_model``; ``team_id`` required)
        - ``recognized``: whether the router knows the name (was ``is_recognized_model``)
        - ``upstream``: the upstream model a name routes to (was ``get_routable_upstream_model``)
        - ``group_of_id``: the group of a deployment id (was ``get_model_group``)
        - ``wildcard_access_group``: whether an access group grants a wildcard route
          (was ``is_model_access_group_for_wildcard_route``)
        - ``team_model_alternatives``: whether a team deployment has siblings to fall back to
          (was ``team_model_has_alternatives``)
        - ``routing_group_alternatives``: (was ``routing_group_has_alternatives``)
        - ``search_tool_support``: per routable deployment, whether it answers Codex's client
          ``tool_search`` (was ``get_search_tool_support``)
        """
        r: Final = self._router
        if kind == "from_id":
            return r._core_resolve_model_name_from_model_id(name, custom_llm_provider=custom_llm_provider)
        if kind == "team_model":
            if team_id is None:
                raise TypeError("registry().resolve(kind='team_model') needs a team_id")
            return r._core_map_team_model(name, team_id, custom_llm_provider=custom_llm_provider)
        if kind == "routing_group_alternatives":
            return r._core_routing_group_has_alternatives(name)
        if name is None:
            raise TypeError(f"registry().resolve(kind={kind!r}) needs a name")
        if kind == "group":
            return r._core_routable_model_group(name)
        if kind == "alias":
            return r._core_get_model_from_alias(model=name)
        if kind == "served":
            # Without a served_model the name is reported as requested, as the core does for a non-match.
            return name if served_model is None else r._core_resolve_served_model_name(name, served_model)
        if kind == "recognized":
            return r._core_is_recognized_model(name)
        if kind == "upstream":
            return r._core_get_routable_upstream_model(name, team_id=team_id)
        if kind == "search_tool_support":
            return r._core_get_search_tool_support(name, team_id=team_id)
        if kind == "group_of_id":
            return r._core_get_model_group(id=name)
        if kind == "wildcard_access_group":
            return r._core_is_model_access_group_for_wildcard_route(model_access_group=name)
        if kind == "team_model_alternatives":
            return r._core_team_model_has_alternatives(name)
        raise ValueError(f"unknown resolve kind {kind!r}")

    def aliases(
        self,
    ) -> "dict[str, str | RouterModelGroupAliasItem]":  # mutable-ok: the model_group_alias attribute, the live dict
        """model_group_alias as configured (was the ``model_group_alias`` attribute)."""
        return self._router._core_model_group_alias

    @overload
    def patterns(self, *, team: Literal[False] = False) -> "PatternMatchRouter": ...

    @overload
    def patterns(self, *, team: Literal[True]) -> "dict[str, PatternMatchRouter]": ...  # mutable-ok: the live dict

    @overload
    def patterns(self, *, team: bool) -> "PatternMatchRouter | dict[str, PatternMatchRouter]": ...  # mutable-ok: same

    def patterns(self, *, team: bool = False) -> "PatternMatchRouter | Mapping[str, PatternMatchRouter]":
        """The wildcard pattern router, or the per-team ones (was ``pattern_router`` / ``team_pattern_routers``)."""
        return self._router._core_team_pattern_routers if team else self._router._core_pattern_router

    # -- groups ---------------------------------------------------------------------------------

    @overload
    def group(
        self,
        name: str,
        *,
        field: Literal["info", "cached_info"] = "info",
        include_hidden: bool = False,
        team_id: str | None = None,
    ) -> "ModelGroupInfo | None": ...

    @overload
    def group(
        self,
        name: str,
        *,
        field: Literal["display_name", "mode"],
        include_hidden: bool = False,
        team_id: str | None = None,
    ) -> str | None: ...

    @overload
    def group(
        self,
        name: str,
        *,
        field: Literal["service_tiers"],
        include_hidden: bool = False,
        team_id: str | None = None,
    ) -> tuple[object, ...]: ...

    @overload
    def group(
        self,
        name: str,
        *,
        field: Literal["token_limits"],
        include_hidden: bool = False,
        team_id: str | None = None,
    ) -> tuple[int | None, int | None]: ...

    @overload
    def group(
        self,
        name: str,
        *,
        field: Literal["listing_info"],
        include_hidden: bool = False,
        team_id: str | None = None,
    ) -> "DeploymentModelListingInfo | None": ...

    def group(
        self,
        name: str,
        *,
        field: GroupField = "info",
        include_hidden: bool = False,
        team_id: str | None = None,
    ) -> (
        "ModelGroupInfo | DeploymentModelListingInfo | tuple[int | None, int | None] | tuple[object, ...] | str | None"
    ):
        """
        What the router knows about a model group. ``field``:

        - ``info``: the group's merged model info (was ``get_model_group_info``)
        - ``cached_info``: (was ``cached_model_group_info``)
        - ``display_name`` / ``mode`` / ``token_limits`` / ``listing_info``: configured values
          (were ``get_configured_display_name``, ``get_configured_mode``,
          ``get_configured_token_limits``, ``get_model_listing_info``)
        - ``service_tiers``: (was ``get_configured_service_tiers``)
        """
        r: Final = self._router
        if field == "info":
            return r._core_get_model_group_info(name, include_hidden=include_hidden, team_id=team_id)
        if field == "cached_info":
            return r._core_cached_model_group_info(name)
        if field == "display_name":
            return r._core_get_configured_display_name(name)
        if field == "mode":
            return r._core_get_configured_mode(name)
        if field == "service_tiers":
            return r._core_get_configured_service_tiers(name, team_id=team_id)
        if field == "token_limits":
            return r._core_get_configured_token_limits(name)
        if field == "listing_info":
            return r._core_get_model_listing_info(name)
        raise ValueError(f"unknown group field {field!r}")

    def routing_groups(self) -> "dict[str, RoutingGroup]":  # mutable-ok: the routing_groups attribute, the live dict
        """Routing groups by name (was the ``routing_groups`` attribute)."""
        return self._router._core_routing_groups

    def access_groups(
        self,
        model_name: str | None = None,
        model_access_group: str | None = None,
        team_id: str | None = None,
    ) -> dict[str, list[str]]:  # mutable-ok: Router._core_get_model_access_groups's type, a fresh dict
        """Access group name to model names (was ``get_model_access_groups``)."""
        return self._router._core_get_model_access_groups(
            model_name=model_name, model_access_group=model_access_group, team_id=team_id
        )

    @overload
    def credentials(
        self, id: str, *, form: Literal["params"] = "params", team_id: str | None = None
    ) -> (
        "UntypedDict | None"
    ): ...  # a mutable list/dict, as before: Router._core_get_deployment_credentials's type, unchanged

    @overload
    def credentials(
        self, id: str, *, form: Literal["with_provider"], team_id: str | None = None
    ) -> "dict[str, Any] | None": ...  # pyright: ignore[reportExplicitAny]  # the core's type, unchanged  # mutable-ok: same

    @overload
    def credentials(
        self, id: str, *, form: Literal["deployment"], team_id: str | None = None
    ) -> "Deployment | None": ...

    def credentials(
        self, id: str, *, form: CredentialForm = "params", team_id: str | None = None
    ) -> "Mapping[str, object] | Deployment | None":
        """
        A deployment's credentials. ``form``: ``params`` (was ``get_deployment_credentials``),
        ``with_provider`` (was ``get_deployment_credentials_with_provider``), ``deployment``, the
        deployment that owns the credential (was ``get_credential_deployment``).
        """
        r: Final = self._router
        if form == "params":
            return r._core_get_deployment_credentials(model_id=id)
        if form == "with_provider":
            return r._core_get_deployment_credentials_with_provider(model_id=id, team_id=team_id)
        if form == "deployment":
            return r._core_get_credential_deployment(model_id=id, team_id=team_id)
        raise ValueError(f"unknown credential form {form!r}")

    @overload
    def plugins(
        self, kind: Literal["auto"]
    ) -> "dict[str, list[TaggedPreRoutingStrategy[AutoRouter]]]": ...  # mutable-ok: the live dict

    @overload
    def plugins(
        self, kind: Literal["complexity"]
    ) -> "dict[str, list[TaggedPreRoutingStrategy[ComplexityRouter]]]": ...  # mutable-ok: the live dict

    @overload
    def plugins(
        self, kind: Literal["quality"]
    ) -> "dict[str, list[TaggedPreRoutingStrategy[QualityRouter]]]": ...  # mutable-ok: the live dict

    @overload
    def plugins(
        self, kind: Literal["adaptive"]
    ) -> "dict[str, list[TaggedPreRoutingStrategy[AdaptiveRouter]]]": ...  # mutable-ok: the live dict

    @overload
    def plugins(self, kind: Literal["routing"]) -> "list[RoutingPlugin]": ...  # mutable-ok: the live list

    @overload
    def plugins(self, kind: Literal["budget_limiter"]) -> "RouterBudgetLimiting | None": ...

    @overload
    def plugins(self, kind: Literal["prompt_caching_pins"]) -> "DualCache": ...

    def plugins(self, kind: PluginKind) -> "RegistryPlugins":
        """
        Plugins configured by the deployments: ``auto``, ``complexity``, ``quality`` and ``adaptive``
        sub-routers by deployment name, ``routing`` plugins, the deployment ``budget_limiter`` and
        the ``prompt_caching_pins`` store (were ``auto_routers``, ``complexity_routers``,
        ``quality_routers``, ``adaptive_routers``, ``routing_plugins``,
        ``get_router_deployment_budget_limiter`` and ``prompt_caching_pin_cache``).
        """
        r: Final = self._router
        if kind == "auto":
            return r._core_auto_routers
        if kind == "complexity":
            return r._core_complexity_routers
        if kind == "quality":
            return r._core_quality_routers
        if kind == "adaptive":
            return r._core_adaptive_routers
        if kind == "routing":
            return r._core_routing_plugins
        if kind == "budget_limiter":
            return r._core_get_router_deployment_budget_limiter()
        if kind == "prompt_caching_pins":
            return r._core_prompt_caching_pin_cache
        raise ValueError(f"unknown plugin kind {kind!r}")


# Settings a caller may read. Assigning one from outside the router goes through apply(ChangeSet(assign=...)).
SETTING_FIELDS: Final[tuple[str, ...]] = (
    "routing_strategy",
    "fallbacks",
    "context_window_fallbacks",
    "content_policy_fallbacks",
    "default_deployment",
    "default_litellm_params",
    "default_max_parallel_requests",
    "cooldown_time",
    "allowed_fails",
    "allowed_fails_policy",
    "disable_cooldowns",
    "enable_pre_call_checks",
    "enable_health_check_routing",
    "health_check_ignore_transient_errors",
    "background_health_check_model_groups",
    "enable_tag_filtering",
    "tag_filtering_match_any",
    "tag_routing_prefix",
    "router_general_settings",
    "provider_budget_config",
    "provider_default_deployment_ids",
    "fallback_access_check",
    "fallback_budget_check",
    "kubernetes_pod_discovery",
    "search_tools",
    "guardrail_list",
    "cache_responses",
)

_MISSING: Final = object()


@dataclass(frozen=True)
class RouterSettings:
    """
    The router's configuration as one frozen snapshot (``settings()``). Fields hold the router's
    own objects, not copies: mutating a list in here mutates the router's.
    A field the router never set reads as ``None``.

    Each field has the type of the Router attribute it reads (``_core_<field>``), the type the old
    ``router.<field>`` attribute had; the mutable ones are the router's live objects. ``of`` fills
    every field. The ``None`` defaults exist for snapshots built directly (tests), which set only the
    fields they need; the fields whose Router attribute cannot be ``None`` say so on their line.
    """

    routing_strategy: str | None = None
    fallbacks: "list | list[dict[str, list[str] | list | None]] | None" = None  # mutable-ok: the router's live list
    context_window_fallbacks: list | None = None  # mutable-ok: the router's live list
    content_policy_fallbacks: list | None = None  # mutable-ok: the router's live list
    default_deployment: object = None
    default_litellm_params: dict = None  # pyright: ignore[reportAssignmentType]  # unset only when built directly (see above)  # mutable-ok: the router's live dict
    default_max_parallel_requests: int | None = None
    cooldown_time: float = None  # pyright: ignore[reportAssignmentType]  # unset only when built directly (see above)
    allowed_fails: int = None  # pyright: ignore[reportAssignmentType]  # unset only when built directly (see above)
    allowed_fails_policy: "AllowedFailsPolicy | None" = None
    disable_cooldowns: bool | None = None
    enable_pre_call_checks: bool = None  # pyright: ignore[reportAssignmentType]  # unset only when built directly (see above)
    enable_health_check_routing: bool = None  # pyright: ignore[reportAssignmentType]  # unset only when built directly (see above)
    health_check_ignore_transient_errors: bool = None  # pyright: ignore[reportAssignmentType]  # unset only when built directly (see above)
    background_health_check_model_groups: frozenset[str] | None = None
    enable_tag_filtering: bool = None  # pyright: ignore[reportAssignmentType]  # unset only when built directly (see above)
    tag_filtering_match_any: bool = None  # pyright: ignore[reportAssignmentType]  # unset only when built directly (see above)
    tag_routing_prefix: str = None  # pyright: ignore[reportAssignmentType]  # unset only when built directly (see above)
    router_general_settings: "RouterGeneralSettings" = None  # pyright: ignore[reportAssignmentType]  # unset only when built directly (see above)
    provider_budget_config: "dict[str, BudgetConfig] | None" = None  # mutable-ok: the router's live dict
    provider_default_deployment_ids: list[str] = None  # pyright: ignore[reportAssignmentType]  # unset only when built directly (see above)  # mutable-ok: the router's live list
    fallback_access_check: "FallbackAccessCheck | None" = None
    fallback_budget_check: "FallbackBudgetCheck | None" = None
    kubernetes_pod_discovery: "KubernetesPodDiscovery" = None  # pyright: ignore[reportAssignmentType]  # unset only when built directly (see above)
    search_tools: "list[SearchToolTypedDict]" = None  # pyright: ignore[reportAssignmentType]  # unset only when built directly (see above)  # mutable-ok: the router's live list
    guardrail_list: "list[GuardrailTypedDict]" = None  # pyright: ignore[reportAssignmentType]  # unset only when built directly (see above)  # mutable-ok: the router's live list
    cache_responses: bool | None = None
    _router: "Router | None" = field(default=None, repr=False, compare=False)
    _unset: frozenset[str] = field(default=frozenset(), repr=False, compare=False)

    @classmethod
    def of(cls, router: "Router") -> "RouterSettings":
        read: Final = {name: getattr(router, "_core_" + name, _MISSING) for name in SETTING_FIELDS}
        values: Final = {name: None if value is _MISSING else value for name, value in read.items()}
        unset: Final = frozenset(name for name, value in read.items() if value is _MISSING)
        # Each value is the Router's own ``_core_<field>`` attribute, typed as the field (or None when unset).
        return cls(**values, _router=router, _unset=unset)  # pyright: ignore[reportArgumentType]  # see above

    def _owner(self, member: str) -> "Router":
        if self._router is None:
            raise AttributeError(f"'NoneType' object has no attribute {member!r}")
        return self._router

    def is_set(self, name: str) -> bool:
        """Whether the router has this setting at all (some exist only in some configurations)."""
        return name in SETTING_FIELDS and name not in self._unset

    def as_dict(self) -> Mapping[str, object]:
        """The runtime-updatable settings and routing groups as plain values (was ``get_settings``)."""
        return self._owner("_core_get_settings")._core_get_settings()

    def allowed_fails_for(self, exception: Exception) -> int | None:
        """Allowed fails for this exception under allowed_fails_policy (was ``get_allowed_fails_from_policy``)."""
        return self._owner("_core_get_allowed_fails_from_policy")._core_get_allowed_fails_from_policy(
            exception=exception
        )

    def guardrail(self, name: str) -> "GuardrailTypedDict":
        """The guardrail configured under this name (was ``get_available_guardrail``)."""
        return self._owner("_core_get_available_guardrail")._core_get_available_guardrail(name)


class RouterHealth:
    """Cooldowns, blocked deployments, health checks and usage (``health()``)."""

    __slots__ = ("_router",)

    def __init__(self, router: "Router") -> None:
        self._router = router

    @property
    def store(self) -> "DualCache":
        """
        The cache that holds routing state (cooldowns, fail counts, usage counters, latency):
        in-memory plus the router's Redis when configured. Read and write it with the router's own
        key names. Replaces the ``cache`` attribute.
        """
        return self._router._core_cache

    @property
    def cooldowns(self) -> "CooldownCache":
        """The cooldown store (was ``cooldown_cache``)."""
        return self._router._core_cooldown_cache

    @property
    def health_checks(self) -> "DeploymentHealthCache":
        """Background health check results (was ``health_state_cache``)."""
        return self._router._core_health_state_cache

    @property
    def latency(
        self,
    ) -> "UntypedDict":  # a mutable list/dict, as before: the deployment_latency_map attribute's type, the live dict
        """Latency samples by deployment (was ``deployment_latency_map``)."""
        return self._router._core_deployment_latency_map

    @staticmethod
    def is_blocked(deployment: "Deployment") -> bool:
        """Whether an admin paused this deployment (was ``is_deployment_blocked``)."""
        from litellm.router import Router

        return Router._core_is_deployment_blocked(deployment)

    @staticmethod
    def all_blocked(deployments: "Sequence[DeploymentTypedDict]") -> bool:
        """Whether every deployment in the list is paused (was ``are_all_deployments_blocked``)."""
        from litellm.router import Router

        return Router._core_are_all_deployments_blocked(deployments)

    def fully_blocked_model_names(self) -> set[str]:  # mutable-ok: the core's type, a fresh set
        """Groups whose every deployment is paused (was ``get_fully_blocked_model_names``)."""
        return self._router._core_get_fully_blocked_model_names()

    async def fully_unhealthy_model_names(self) -> set[str]:  # mutable-ok: the core's type, a fresh set
        """Groups whose every deployment failed its health check (was ``async_get_fully_unhealthy_model_names``)."""
        return await self._router._core_async_get_fully_unhealthy_model_names()

    @overload
    async def usage(
        self, model_group: str, *, kind: Literal["current"] = "current"
    ) -> tuple[int | None, int | None]: ...

    @overload
    async def usage(
        self, model_group: str, *, kind: Literal["remaining"]
    ) -> dict[str, int]: ...  # mutable-ok: Router._core_get_remaining_model_group_usage's type, a fresh dict

    async def usage(
        self, model_group: str, *, kind: UsageKind = "current"
    ) -> tuple[int | None, int | None] | Mapping[str, int]:
        """
        tpm/rpm usage of a group this minute (``current``, was ``get_model_group_usage``) or what is
        left of its limits (``remaining``, was ``get_remaining_model_group_usage``).
        """
        if kind == "current":
            return await self._router._core_get_model_group_usage(model_group=model_group)
        if kind == "remaining":
            return await self._router._core_get_remaining_model_group_usage(model_group=model_group)
        raise ValueError(f"unknown usage kind {kind!r}")

    def team_ptu_ceiling(self, team_id: str, model_group: str) -> "PTUTeamCeiling | None":
        """A team's PTU share ceiling on a group (was ``team_ptu_ceiling``)."""
        return self._router._core_team_ptu_ceiling(team_id, model_group)

    def split_priority(self, model: str, kwargs: Mapping[str, object]) -> tuple[int | None, Mapping[str, object]]:
        """The request priority and the kwargs without it (was ``split_request_priority``)."""
        return self._router._core_split_request_priority(model, kwargs)


@dataclass
class ChangeSet:
    """
    Runtime changes for ``apply``. Applied in this order: ``assign``, ``settings``, ``upsert``,
    ``delete``, ``redis_cache``.

    - ``assign``: replace settings fields as they are (what ``router.<field> = value`` did)
    - ``settings``: validated runtime settings (what ``update_settings(**settings)`` did)
    - ``upsert``: add or replace deployments (``upsert_deployment``)
    - ``delete``: remove deployments by id (``delete_deployment``)
    - ``redis_cache``: point the routing state at a new Redis cache (``update_redis_cache``)
    - ``refresh_model_info``: re-read discovered model info (``arefresh_model_info``; ``aapply`` only)
    """

    assign: Mapping[str, object] = field(default_factory=dict)
    settings: Mapping[str, object] = field(default_factory=dict)
    upsert: Sequence["Deployment"] = ()
    delete: Sequence[str] = ()
    redis_cache: "RedisCache | None" = None
    refresh_model_info: bool = False
    refresh_client: "AsyncHTTPHandler | None" = None


@dataclass
class ApplyResult:
    """What ``apply`` did: the registry version after it, and per upsert/delete the old deployment."""

    version: int
    upserted: "Sequence[Deployment | None]" = field(default_factory=list)
    deleted: "Sequence[Deployment | None]" = field(default_factory=list)


@dataclass(frozen=True)
class RoutingObservers:
    """The router's routing observers (``observe()``)."""

    slack_alerting_logger: "SlackAlerting | None" = None


def apply_changes(router: "Router", changes: ChangeSet) -> ApplyResult:
    if changes.refresh_model_info:
        raise TypeError("ChangeSet(refresh_model_info=True) needs aapply()")
    unknown: Final = sorted(set(changes.assign) - set(SETTING_FIELDS))
    if unknown:
        raise ValueError(f"not assignable router settings: {unknown}")
    for name, value in changes.assign.items():
        setattr(router, "_core_" + name, value)
    if changes.settings:
        router._core_update_settings(**changes.settings)
    upserted: Final = [router._core_upsert_deployment(deployment=d) for d in changes.upsert]
    deleted: Final = [router._core_delete_deployment(id=i) for i in changes.delete]
    if changes.redis_cache is not None:
        router._core_update_redis_cache(cache=changes.redis_cache)
    return ApplyResult(version=router._registry_version, upserted=upserted, deleted=deleted)


async def aapply_changes(router: "Router", changes: ChangeSet) -> ApplyResult:
    refresh: Final = changes.refresh_model_info
    result: Final = apply_changes(
        router,
        ChangeSet(
            assign=changes.assign,
            settings=changes.settings,
            upsert=changes.upsert,
            delete=changes.delete,
            redis_cache=changes.redis_cache,
        ),
    )
    if refresh:
        await router._core_arefresh_model_info(client=changes.refresh_client)
    return result


class RoutingContext:
    """
    What a routing plugin (strategy, pre-call check, sub-router) may use of the router. Plugins get
    one instead of the Router itself.
    """

    __slots__ = ("_router",)

    def __init__(self, router: "Router") -> None:
        self._router = router

    def registry(self) -> RouterRegistry:
        return self._router.registry()

    def settings(self) -> RouterSettings:
        return self._router.settings()

    def health(self) -> RouterHealth:
        return self._router.health()

    def pre_call_checks(
        self,
        model: str,
        healthy_deployments: list[object],  # mutable-ok: Router._pre_call_checks takes and filters a list
        messages: list[dict[str, str]] | None = None,  # mutable-ok: Router._pre_call_checks takes a list
        input: str | list[object] | None = None,  # mutable-ok: Router._pre_call_checks takes a list
        request_kwargs: dict[str, object] | None = None,  # mutable-ok: Router._pre_call_checks takes a dict
        input_token_count: int | None = None,
        skip_inline_token_count: bool = False,
        request_deployments: "Sequence[DeploymentTypedDict] | None" = None,
    ) -> Sequence[object]:
        return self._router._pre_call_checks(
            model=model,
            healthy_deployments=healthy_deployments,
            messages=messages,
            input=input,
            request_kwargs=request_kwargs,
            input_token_count=input_token_count,
            skip_inline_token_count=skip_inline_token_count,
            request_deployments=request_deployments,
        )

    def count_pre_call_check_tokens(
        self,
        messages: Sequence[Mapping[str, object]] | None,
        input: str | list[object] | None,  # mutable-ok: Router._count_pre_call_check_tokens takes a list
        request_kwargs: Mapping[str, object] | None = None,
    ) -> int:
        return self._router._count_pre_call_check_tokens(messages, input, request_kwargs)

    def update_kwargs_before_fallbacks(
        self,
        model: str,
        kwargs: dict[str, object],  # mutable-ok: the core updates the request kwargs in place
        metadata_variable_name: str | None = "metadata",
    ) -> None:
        return self._router._update_kwargs_before_fallbacks(model, kwargs, metadata_variable_name)

    def arm_read_prefetch(
        self,
        model: str,
        request_kwargs: dict[str, object] | None = None,  # mutable-ok: the core takes a dict
    ) -> None:
        """(was ``arm_routing_read_prefetch``)"""
        return self._router._core_arm_routing_read_prefetch(model, request_kwargs)

    async def pre_routing_hook(
        self,
        model: str,
        request_kwargs: dict[str, object],  # mutable-ok: Router._core_async_pre_routing_hook takes a dict
        messages: list[dict[str, object]] | None = None,  # mutable-ok: the core takes a list
        input: str | list[object] | None = None,  # mutable-ok: Router._core_async_pre_routing_hook takes a list
        specific_deployment: bool | None = False,
    ) -> "PreRoutingHookResponse | None":
        """(was ``async_pre_routing_hook``)"""
        return await self._router._core_async_pre_routing_hook(
            model=model,
            request_kwargs=request_kwargs,
            messages=messages,
            input=input,
            specific_deployment=specific_deployment,
        )

    def log_retry(
        self,
        kwargs: dict[str, object],  # mutable-ok: Router._core_log_retry records the retry into the request kwargs
        e: Exception,
    ) -> Mapping[str, object]:
        return self._router._core_log_retry(kwargs=kwargs, e=e)


__all__ = [
    "SETTING_FIELDS",
    "ApplyResult",
    "ChangeSet",
    "RouterHealth",
    "RouterRegistry",
    "RouterSettings",
    "RoutingContext",
    "RoutingObservers",
    "deployment_id",
]
