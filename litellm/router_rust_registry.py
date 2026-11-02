# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAny=false
# (the ported core code reads untyped litellm helpers: PatternMatchRouter, Deployment.to_json, host.invoke)
"""
``HostRegistry``: the Rust core's live ``model_list`` and the indexes ``registry()`` reads.

Bridged host-side, to port to router-core: building a deployment needs live Python objects
(``get_secret`` for ``os.environ/`` references, ``get_llm_provider`` for provider validation,
``litellm.register_model`` for the pricing each deployment writes into ``litellm.model_cost``), and
the lookups read the stored entries those produce. The rules are the Python router core's, ported
here (never imported from it): the same entries, the same index positions and the same answers.
The Rust core routes over the entries this registry holds; ``RustCore`` hands it every change.

The attribute and method names follow the router core's (``_core_model_list``,
``_core_get_model_list`` ...), so ``litellm.router_api.RouterRegistry`` reads this object as its
router and every ``registry()`` query dispatches the way it does on the Python core.
"""

import copy
import re
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from functools import lru_cache, partial
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, TypeAlias

from pydantic import TypeAdapter

from litellm import exceptions as litellm_exceptions
from litellm import get_secret_str
from litellm._internal_context import service_target
from litellm._logging import verbose_router_logger
from litellm.caching.dual_cache import DualCache
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.constants import DEFAULT_MAX_LRU_CACHE_SIZE
from litellm.litellm_core_utils.access_windows import access_windows_config_error
from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.litellm_core_utils.credential_accessor import CredentialAccessor
from litellm.litellm_core_utils.get_llm_provider_logic import (
    declared_authenticating_provider,
    is_registered_custom_provider,
    resolve_model_and_provider_for_metadata,
)
from litellm.litellm_core_utils.llm_cost_calc.utils import SERVICE_TIER_COST_KEY_SUFFIXES
from litellm.litellm_core_utils.pattern_match_deployments import PatternMatchRouter
from litellm.litellm_core_utils.ptu_pricing import (
    PTU_COST_ATTRIBUTION_ENV_VAR,
    declares_ptu,
    declares_ptu_shares,
    is_ptu_cost_attribution_enabled,
    ptu_config_error,
    ptu_identity_error,
    ptu_terms,
    zeroed_ptu_pricing,
)
from litellm.litellm_core_utils.ptu_shares import ptu_capacity_warning
from litellm.litellm_core_utils.routing_common_utils import (
    provider_for_generic_call,
    resolve_model_group_alias,
    team_may_use_deployment,
    warn_on_ignored_model_capability_flags,
    warn_on_provider_credential_mismatch,
    with_admin_api_key_secret_reference,
    without_model_capability_flags,
)
from litellm.litellm_core_utils.routing_groups import apply_routing_group_priority
from litellm.llms.openai_like.json_loader import JSONProviderRegistry
from litellm.llms.openai_like.model_info import MODEL_INFO_REFRESH_SECONDS
from litellm.responses.bridge_check import keep_registered_mode
from litellm.router_api import RouterRegistry, deployment_id
from litellm.router_core import organization_deployment_id
from litellm.router_rust_model_info import HostModelInfo
from litellm.router_rust_shaping import ROUTER_CLIENTS_TARGET
from litellm.router_strategy.auto_router.auto_router_model_naming import classify_strategy_router_model
from litellm.router_utils.endpoint_eligibility import warn_on_unenforced_endpoint_flag
from litellm.types.router import (
    VALID_LITELLM_ENVIRONMENTS,
    CredentialLiteLLMParams,
    Deployment,
    DeploymentTypedDict,
    LiteLLM_Params,
    ModelGroupInfo,
    RouterModelGroupAliasItem,
    RoutingGroup,
    RoutingPlugin,
    TaggedPreRoutingStrategy,
    holds_secret_pointer,
    validate_litellm_params_strictly,
)
from litellm.types.utils import (
    CustomPricingLiteLLMParams,
    LlmProviders,
    is_server_derived_pricing_key,
    shared_backend_model_info,
)
from litellm.utils import (
    ModelInfoLookupMiss,
    get_secret,
    lookup_model_info_for_registration,
    model_info_with_pricing_overrides,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.ptu_shares import PTUTeamCeiling
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
    from litellm.router_python_host import PythonHost
    from litellm.router_rust_callbacks import OptionalChecks
    from litellm.router_rust_prerouting import PreRoutingStrategies
    from litellm.router_strategy.budget_limiter import RouterBudgetLimiting

# The PTU share teams and the team ceilings read so far.
_PTUMemo: TypeAlias = tuple[
    frozenset[str],
    dict[tuple[str, str], "PTUTeamCeiling | None"],
]
_MODEL_INFO_ADAPTER: Final = TypeAdapter(Mapping[str, object])
_LITELLM_PARAMS_ADAPTER: Final = TypeAdapter(Mapping[str, object])


_CACHE_PRICING_FIELDS: Final = (
    "cache_creation_input_token_cost",
    "cache_creation_input_token_cost_above_1hr",
    "cache_creation_input_token_cost_above_200k_tokens",
    "cache_read_input_token_cost",
    "cache_read_input_token_cost_above_200k_tokens",
)
_OFF_PEAK_BASE_RATE_FIELDS: Final = (
    "input_cost_per_token",
    "input_cost_per_second",
    "cost_per_second",
    "tiered_pricing",
)

# A stored model_list entry: the live dict the router core keeps.
UntypedEntry: TypeAlias = dict[str, object]


def model_info_is_active_for_environment(model_info: Mapping[str, object] | None) -> bool:
    """The environment-gating rule (``model_info_is_active_for_environment``): a deployment whose
    model_info names ``supported_environments`` loads only where LITELLM_ENVIRONMENT is in it."""
    if model_info is None:
        return True
    supported_environments: Final = model_info.get("supported_environments")
    if supported_environments is None:
        return True
    if not isinstance(supported_environments, list | tuple):
        raise ValueError(  # noqa: TRY004  # the core raises ValueError here
            f"supported_environments must be a list of {VALID_LITELLM_ENVIRONMENTS}. "
            f"but set as: {supported_environments} for model_info: {model_info}"
        )
    litellm_environment: Final = get_secret_str(secret_name="LITELLM_ENVIRONMENT")
    if litellm_environment is None:
        raise ValueError("Set 'supported_environments' for model but not 'LITELLM_ENVIRONMENT' set in .env")
    if litellm_environment not in VALID_LITELLM_ENVIRONMENTS:
        raise ValueError(
            f"LITELLM_ENVIRONMENT must be one of {VALID_LITELLM_ENVIRONMENTS}. but set as: {litellm_environment}"
        )
    for environment in supported_environments:  # pyright: ignore[reportUnknownVariableType]  # a configured list
        if environment not in VALID_LITELLM_ENVIRONMENTS:
            raise ValueError(
                f"supported_environments must be one of {VALID_LITELLM_ENVIRONMENTS}. but set as: {environment} "
                f"for model_info: {model_info}"
            )
    return litellm_environment in supported_environments


# -- pricing written into litellm.model_cost -----------------------------------------------------


def _backend_model_info(
    host: "PythonHost", backend_model: str, custom_llm_provider: str | None
) -> Mapping[str, object]:
    """``get_model_info`` for the backend model; raises like it for an unmapped model."""
    info: Final = host.invoke("get_model_info")(model=backend_model, custom_llm_provider=custom_llm_provider)
    return _MODEL_INFO_ADAPTER.validate_python(info)


def _backend_cost_entry(
    host: "PythonHost", backend_model: str, custom_llm_provider: str | None
) -> Mapping[str, object] | None:
    """The backend model's raw ``litellm.model_cost`` entry, or None when it has none."""
    try:
        info: Final = _backend_model_info(host, backend_model, custom_llm_provider)
    except Exception:  # noqa: BLE001  # get_model_info raises plain Exception for an unmapped backend model
        return None
    entry: Final[object] = host.dynamic().model_cost.get(str(info.get("key") or ""))
    return _MODEL_INFO_ADAPTER.validate_python(entry) if isinstance(entry, dict) else None


def _inherited_cache_pricing(
    host: "PythonHost", model_info: Mapping[str, object], backend_model: str, provider: str | None
) -> Mapping[str, object]:
    """The cache prices a custom-priced entry inherits (``_inherit_builtin_cache_pricing``)."""
    inherited_fields: Final = (
        _CACHE_PRICING_FIELDS
        if model_info.get("cache_read_input_token_cost") is not None
        else (*_CACHE_PRICING_FIELDS, "explicit_cache_read_input_token_cost")
    )
    missing_fields: Final = tuple(field for field in inherited_fields if model_info.get(field) is None)
    if not missing_fields:
        return {}
    try:
        backend_info: Final = _backend_model_info(host, backend_model, provider)
    except Exception:  # noqa: BLE001  # an unmapped backend model contributes nothing
        return {}
    return {field: backend_info[field] for field in missing_fields if backend_info.get(field) is not None}


def _inherited_service_tier_pricing(
    host: "PythonHost", model_info: Mapping[str, object], backend_model: str, provider: str | None
) -> Mapping[str, object]:
    """The service-tier rates a custom-priced entry inherits (``_inherit_builtin_service_tier_pricing``)."""
    if ptu_terms(model_info) is not None and is_ptu_cost_attribution_enabled():
        return {}
    if all(model_info.get(field) is None for field in ("input_cost_per_token", "output_cost_per_token")):
        return {}
    backend_entry: Final = _backend_cost_entry(host, backend_model, provider)
    if backend_entry is None:
        return {}
    return {
        field: copy.deepcopy(value)
        for field, value in backend_entry.items()
        if field.endswith(SERVICE_TIER_COST_KEY_SUFFIXES) and model_info.get(field) is None and value is not None
    }


def _inherited_off_peak_base_rates(
    host: "PythonHost", model_info: Mapping[str, object], backend_model: str, provider: str | None
) -> Mapping[str, object]:
    """The whole backend entry an off-peak-only entry copies (``_inherit_builtin_base_rates_for_off_peak``)."""
    if not model_info.get("off_peak_pricing"):
        return {}
    if any(model_info.get(field) is not None for field in _OFF_PEAK_BASE_RATE_FIELDS):
        return {}
    backend_entry: Final = _backend_cost_entry(host, backend_model, provider)
    if backend_entry is None:
        return {}
    return {
        field: copy.deepcopy(value)
        for field, value in backend_entry.items()
        if model_info.get(field) is None and value is not None
    }


def _inherited_tiered_output_rate(
    host: "PythonHost", model_info: Mapping[str, object], backend_model: str, provider: str | None
) -> Mapping[str, object]:
    """The flat output rate a tier table without one inherits (``_inherit_builtin_tiered_output_rate``)."""
    tiers: Final = model_info.get("tiered_pricing")
    if not isinstance(tiers, list) or not tiers:
        return {}
    if model_info.get("output_cost_per_token") is not None:
        return {}
    if all(isinstance(tier, dict) and "output_cost_per_token" in tier for tier in tiers):  # pyright: ignore[reportUnknownVariableType]  # a configured tier table
        return {}
    try:
        backend_info: Final = _backend_model_info(host, backend_model, provider)
    except Exception:  # noqa: BLE001  # an unmapped backend model contributes nothing
        return {}
    backend_rate: Final = backend_info.get("output_cost_per_token")
    return {"output_cost_per_token": backend_rate} if backend_rate else {}


def _bedrock_profile(params: Mapping[str, object], model: str) -> bool:
    return params.get("custom_llm_provider") == "bedrock" or model.startswith("bedrock/")


def _deployment_ids(deployments: Iterable[Mapping[str, object]]) -> frozenset[str]:
    return frozenset(
        str(model_info["id"])
        for deployment in deployments
        if (model_info := _model_info_of(deployment)).get("id") is not None
    )


def _search_tool_support(deployment: Mapping[str, object]) -> bool | None:
    """A deployment's answer to Codex's client ``tool_search``."""
    from litellm.responses.litellm_completion_transformation.tool_search import chat_config_builds_anthropic_request
    from litellm.utils import ProviderConfigManager

    configured: Final = _model_info_of(deployment).get("supports_search_tool")
    if isinstance(configured, bool):
        return configured
    params: Final = _params_of(deployment)
    model: Final = params.get("model")
    provider: Final = params.get("custom_llm_provider")
    if not isinstance(model, str):
        return None
    try:
        resolved_model, resolved_provider = resolve_model_and_provider_for_metadata(
            model=model, custom_llm_provider=provider if isinstance(provider, str) else None
        )
        chat_config: Final = ProviderConfigManager.get_provider_chat_config(
            model=resolved_model, provider=LlmProviders(resolved_provider)
        )
    except (litellm_exceptions.BadRequestError, ValueError):
        return None
    return True if chat_config_builds_anthropic_request(chat_config) else None


def cost_map_backend_model(deployment: Deployment) -> str:
    """The model a deployment's prices are inherited from (``_cost_map_backend_model``)."""
    model_info_base_model: Final = deployment.model_info.base_model
    if isinstance(model_info_base_model, str) and model_info_base_model:
        return model_info_base_model
    params: Final = _LITELLM_PARAMS_ADAPTER.validate_python(deployment.litellm_params.model_dump(exclude_none=True))
    params_base_model: Final = params.get("base_model")
    if isinstance(params_base_model, str) and params_base_model:
        return params_base_model
    model: Final = deployment.litellm_params.model
    invoked_model: Final = params.get("model_id")
    if isinstance(invoked_model, str) and invoked_model and _bedrock_profile(params, model):
        profile_model: Final = invoked_model if invoked_model.startswith("bedrock/") else f"bedrock/{invoked_model}"
        profile_info: Final = lookup_model_info_for_registration(model=profile_model, custom_llm_provider="bedrock")
        if not isinstance(profile_info, ModelInfoLookupMiss):
            return profile_model
    return model


def deployment_model_cost_payload(host: "PythonHost", deployment: Deployment) -> Mapping[str, object]:
    """The ``model_info`` a deployment contributes to ``litellm.model_cost``
    (``_deployment_model_cost_payload``): its model_info, its custom pricing, then the inherited rates."""
    declared: Final = dict(_MODEL_INFO_ADAPTER.validate_python(deployment.model_info.model_dump(exclude_none=True)))
    custom_pricing: Final = {
        field: value
        for field in CustomPricingLiteLLMParams.model_fields
        if (value := deployment.litellm_params.get(field)) is not None
    }
    backend_model: Final = cost_map_backend_model(deployment)
    provider: Final = deployment.model_info.pricing_provider or deployment.litellm_params.custom_llm_provider
    params: Final = _LITELLM_PARAMS_ADAPTER.validate_python(deployment.litellm_params.model_dump(exclude_none=True))
    priced: Final = {**declared, **custom_pricing}
    profile_pricing: Final[Mapping[str, object]] = (
        _bedrock_profile_pricing(backend_model, priced)
        if params.get("model_id")
        and backend_model != deployment.litellm_params.model
        and _bedrock_profile(params, deployment.litellm_params.model)
        else {}
    )
    with_profile: Final = {**priced, **profile_pricing}
    off_peak: Final = {**with_profile, **_inherited_off_peak_base_rates(host, with_profile, backend_model, provider)}
    cached: Final = (
        {**off_peak, **_inherited_cache_pricing(host, off_peak, backend_model, provider)}
        if off_peak.get("input_cost_per_token") is not None
        else off_peak
    )
    tiers: Final = {**cached, **_inherited_service_tier_pricing(host, cached, backend_model, provider)}
    return {**tiers, **_inherited_tiered_output_rate(host, tiers, backend_model, provider)}


def _bedrock_profile_pricing(backend_model: str, priced: Mapping[str, object]) -> Mapping[str, object]:
    profile_info: Final = lookup_model_info_for_registration(model=backend_model, custom_llm_provider="bedrock")
    if isinstance(profile_info, ModelInfoLookupMiss):
        return {}
    explicit_info: Final = _MODEL_INFO_ADAPTER.validate_python(priced)
    priced_info: Final = model_info_with_pricing_overrides(profile_info, explicit_info)  # pyright: ignore[reportArgumentType]  # the explicit model_info, as the core passes it
    return {key: value for key, value in priced_info.items() if is_server_derived_pricing_key(key)}


def _backend_cost_map_keys(model: str, custom_llm_provider: str | None) -> tuple[str, ...]:
    """The ``litellm.model_cost`` keys a deployment's shared backend info is registered under."""
    backend_key: Final = model if custom_llm_provider is None else f"{custom_llm_provider}/{model}"
    if "responses/" in backend_key:
        return (backend_key, backend_key.replace("responses/", ""))
    return (backend_key,)


def register_deployment_in_model_cost(
    host: "PythonHost",
    *,
    model_id: str | None,
    model_info: Mapping[str, object],
    model: str,
    custom_llm_provider: str | None,
    pricing_provider: str | None = None,
) -> None:
    """Write a deployment's metadata into ``litellm.model_cost`` (``_register_deployment_in_model_cost``):
    the whole entry under its id, the schema fields under its backend model key(s)."""
    registered: Final = (
        {key: value for key, value in model_info.items() if key not in CustomPricingLiteLLMParams.model_fields}
        if classify_strategy_router_model(model) is not None
        else dict(model_info)
    )
    register_model: Final = host.invoke("register_model")
    if model_id is not None:
        register_model(
            model_cost={model_id: registered},
            persist_across_reloads=False,
            warning_display_name=model,
            custom_llm_provider=pricing_provider or custom_llm_provider,
            add_to_provider_model_sets=False,
        )
    shared_model_info: Final = shared_backend_model_info(registered)
    register_model(
        model_cost={
            key: keep_registered_mode(key, shared_model_info)
            for key in _backend_cost_map_keys(model=model, custom_llm_provider=custom_llm_provider)
        },
        persist_across_reloads=False,
    )


def register_deployment_pricing(host: "PythonHost", deployment: Deployment) -> None:
    """Register a deployment's custom and inherited pricing (``_register_deployment_pricing``)."""
    register_deployment_in_model_cost(
        host,
        model_id=deployment.model_info.id,
        model_info=deployment_model_cost_payload(host, deployment),
        model=deployment.litellm_params.model,
        custom_llm_provider=deployment.litellm_params.custom_llm_provider,
        pricing_provider=deployment.model_info.pricing_provider,
    )


# -- the registry --------------------------------------------------------------------------------


def _model_info_of(entry: Mapping[str, object]) -> Mapping[str, object]:
    model_info: Final[object] = entry.get("model_info")
    return _MODEL_INFO_ADAPTER.validate_python(model_info) if isinstance(model_info, dict) else {}


def _params_of(entry: Mapping[str, object]) -> Mapping[str, object]:
    params: Final[object] = entry.get("litellm_params")
    return _LITELLM_PARAMS_ADAPTER.validate_python(params) if isinstance(params, dict) else {}


def _renamed(entry: UntypedEntry, model_name: str) -> UntypedEntry:
    return {**entry, "model_name": model_name}


def _as_routing_group_row(deployment: Mapping[str, object]) -> UntypedEntry:
    """A member deployment under its group's name, without the member's ``access_groups``."""
    model_info: Final = {k: v for k, v in _model_info_of(deployment).items() if k != "access_groups"}
    return {**deployment, "model_info": model_info}


def _without_indices(indices: Sequence[int], removal_idx: int) -> list[int]:  # mutable-ok: an index list
    return [idx - 1 if idx > removal_idx else idx for idx in indices if idx != removal_idx]


class HostRegistry(HostModelInfo):
    """The live model_list and its indexes, kept the way the Python router core keeps them."""

    def __init__(
        self,
        host: "PythonHost",
        *,
        model_group_alias: "dict[str, str | RouterModelGroupAliasItem]",  # mutable-ok: the router's live alias map
        ignore_invalid_deployments: bool,
        on_change: Callable[[], None],
    ) -> None:
        self._host: Final = host
        self._on_change: Final = on_change
        self.ignore_invalid_deployments: bool = ignore_invalid_deployments
        self._registry_version: int = 0
        self._deployment_index_lock: Final = threading.RLock()
        self._core_model_group_alias: dict[str, str | RouterModelGroupAliasItem] = model_group_alias  # mutable-ok: the live alias map  # fmt: skip
        self._core_model_list: list[UntypedEntry] = []  # mutable-ok: the live model_list
        self.model_id_to_deployment_index_map: dict[str, int] = {}  # mutable-ok: the live id index
        self._core_model_name_to_deployment_indices: dict[str, list[int]] = {}  # mutable-ok: the live name index
        self._core_team_model_to_deployment_indices: dict[tuple[str, str], list[int]] = {}  # mutable-ok: the live team index  # fmt: skip
        self._core_team_public_model_names: frozenset[str] = frozenset()
        self._core_team_pattern_routers: dict[str, PatternMatchRouter] = {}  # mutable-ok: the live per-team routers
        self._core_pattern_router: PatternMatchRouter = PatternMatchRouter()
        self._core_provider_default_deployment_ids: list[str] = []  # mutable-ok: the live id list
        self._core_deployment_names: list[str] = []  # mutable-ok: the live name list
        self._core_model_names: set[str] = set()  # mutable-ok: the live group set
        self._core_deployment_latency_map: dict[str, float] = {}  # mutable-ok: the live latency map
        self._access_groups_cache: Mapping[str, Sequence[str]] | None = None
        # The routing groups by name; RustCore sets them from its resolved settings.
        self._core_routing_groups: dict[str, RoutingGroup] = {}  # mutable-ok: the routing_groups attribute
        self._routing_group_rows: tuple[UntypedEntry, ...] | None = None
        self._provider_unresolved_deployments: tuple[Callable[[], Deployment | None], ...] = ()
        # The pre-routing strategy registries (``registry().plugins``) and the hooks that keep them;
        # RustCore installs its ``PreRoutingStrategies`` (litellm/router_rust_prerouting.py).
        self._core_auto_routers: dict[str, list[TaggedPreRoutingStrategy[object]]] = {}  # mutable-ok: the live registry
        self._core_complexity_routers: dict[str, list[TaggedPreRoutingStrategy[object]]] = {}  # mutable-ok: same
        self._core_adaptive_routers: dict[str, list[TaggedPreRoutingStrategy[object]]] = {}  # mutable-ok: same
        self._core_quality_routers: dict[str, list[TaggedPreRoutingStrategy[object]]] = {}  # mutable-ok: same
        self.strategies: PreRoutingStrategies | None = None
        # The router's optional pre-call checks (deployment budgets, prompt caching pins);
        # RustCore installs its ``OptionalChecks`` (litellm/router_rust_callbacks.py).
        self.checks: OptionalChecks | None = None
        # The Rust core's resolution for a request (`_core_get_routable_deployments`), set by the core.
        self.routable_lookup: Callable[[str, Mapping[str, object]], Sequence[Mapping[str, object]]] | None = None
        # A deployment's cached `max_parallel_requests` limit (litellm/router_rust_slots.py), set by the core.
        self.max_parallel_requests_lookup: Callable[[Mapping[str, object], Mapping[str, object]], object] | None = None
        # `plugins("routing")`: the router's routing plugins, set by the core.
        self._core_routing_plugins: list[RoutingPlugin] = []  # mutable-ok: the Python core hands out its live list
        # `zero_cost()`: the per-deployment zero-cost lookup the auth checks fill, emptied with the
        # group info caches.
        self._zero_cost_cache: dict[str, bool] = {}  # mutable-ok: the auth checks write into it
        # The routing cache usage reads come from; RustCore sets it to the one its Rust core writes.
        self._core_cache: DualCache = DualCache()
        self._discovered_model_info_cache: InMemoryCache = InMemoryCache(
            max_size_in_memory=1, default_ttl=2 * MODEL_INFO_REFRESH_SECONDS
        )
        # The PTU share teams and team ceilings, until the model list next changes.
        self._ptu_cache: _PTUMemo | None = None
        self._cached_get_model_group_info: Callable[[str], ModelGroupInfo | None] = lru_cache(  # pyright: ignore[reportIncompatibleMethodOverride]  # the per-router cache shadows the uncached method, as the core keeps it
            maxsize=DEFAULT_MAX_LRU_CACHE_SIZE
        )(self._core_get_model_group_info)
        self._core_cached_deployment_model_info: Callable[[str, str], object] = lru_cache(
            maxsize=DEFAULT_MAX_LRU_CACHE_SIZE
        )(self._core_get_deployment_model_info)

    def __getattr__(self, name: str) -> object:
        from litellm.router_rust_core import RustCoreUnsupported

        raise RustCoreUnsupported(f"registry().{name.removeprefix('_core_')} is not served by the Rust router core")

    # -- building ---------------------------------------------------------------------------------

    def build(self, model_list: Sequence[object] | None) -> None:
        """The constructor's registry: ``set_model_list`` and the latency map, as the core builds them."""
        if model_list is None:
            return
        self._discovered_model_info_cache = InMemoryCache(
            max_size_in_memory=max(len(model_list), 1), default_ttl=2 * MODEL_INFO_REFRESH_SECONDS
        )
        self.set_model_list(model_list)
        for entry in model_list:
            params = _params_of(_LITELLM_PARAMS_ADAPTER.validate_python(entry))
            if "model" in params:
                self._core_deployment_latency_map[str(params["model"])] = 0

    async def arefresh_model_info(self, client: "AsyncHTTPHandler | None") -> None:
        """``_core_arefresh_model_info``: token limits the served OpenAI-compatible deployments
        advertise on their models endpoint, read with a few workers over a snapshot of the list."""
        import asyncio

        from litellm.llms.openai_like.model_info import MODEL_INFO_REFRESH_CONCURRENCY

        deployments: Final = iter(tuple(self._core_model_list))

        async def refresh_worker() -> None:
            for raw_deployment in deployments:
                try:
                    await self._arefresh_deployment_model_info(raw_deployment, client=client)
                except Exception:  # noqa: BLE001  # one invalid deployment must not prevent refreshing the others
                    verbose_router_logger.debug("Could not refresh deployment model info")

        _ = await asyncio.gather(*(refresh_worker() for _ in range(MODEL_INFO_REFRESH_CONCURRENCY)))
        self._invalidate_model_group_info_cache()

    async def _arefresh_deployment_model_info(
        self, raw_deployment: Mapping[str, object], *, client: "AsyncHTTPHandler | None"
    ) -> None:
        from litellm.llms.custom_httpx.http_handler import get_async_httpx_client
        from litellm.llms.openai_like.model_info import (
            MODEL_INFO_DISCOVERY_PROVIDERS,
            get_openai_compatible_model_info,
        )
        from litellm.types.router import DiscoveredDeploymentModelInfo
        from litellm.utils import aresolve_api_key_secret_reference

        deployment: Final = Deployment.model_validate(raw_deployment)
        params: Final = LiteLLM_Params.model_validate(
            MappingProxyType(
                {
                    **deployment.litellm_params.model_dump(exclude_none=True),
                    **(
                        self._core_get_deployment_credentials_with_provider(deployment.model_info.id or "")
                        or MappingProxyType({})
                    ),
                }
            )
        )
        model, provider, dynamic_api_key, api_base = self._host.invoke("get_llm_provider")(  # pyright: ignore[reportAny]  # litellm's untyped tuple
            model=params.model, litellm_params=params
        )
        if provider not in MODEL_INFO_DISCOVERY_PROVIDERS:
            return
        if api_base is None or "*" in model or params.get("use_clientside_credentials"):  # pyright: ignore[reportUnknownMemberType]  # the params' untyped get
            return
        api_key: Final = await aresolve_api_key_secret_reference(
            params.api_key or dynamic_api_key,  # pyright: ignore[reportAny]  # see above
            model=model,  # pyright: ignore[reportAny]  # see above
            llm_provider=provider,  # pyright: ignore[reportAny]  # see above
            upstream=MappingProxyType({"api_base": params.api_base}),
        )
        headers: Final = TypeAdapter(Mapping[str, str]).validate_python(
            params.get("extra_headers") or params.get("headers") or MappingProxyType({})  # pyright: ignore[reportUnknownMemberType]  # see above
        )
        auth_headers: Final = (
            MappingProxyType({"authorization": f"Bearer {api_key}"}) if api_key else MappingProxyType({})
        )
        limits: Final = await get_openai_compatible_model_info(
            model=model,  # pyright: ignore[reportAny]  # see above
            api_base=api_base,  # pyright: ignore[reportAny]  # see above
            headers=MappingProxyType(
                {**auth_headers, **MappingProxyType({key.lower(): value for key, value in headers.items()})}
            ),
            client=client or get_async_httpx_client(llm_provider=LlmProviders.OPENAI),
            cache=self._core_cache.in_memory_cache,
        )
        model_id: Final = deployment.model_info.id
        if not limits or model_id is None or self._core_get_model_info(model_id) is not raw_deployment:
            return
        self._discovered_model_info_cache.max_size_in_memory = max(len(self._core_model_list), 1)
        self._discovered_model_info_cache.delete_cache(model_id)
        self._discovered_model_info_cache.set_cache(
            model_id,
            DiscoveredDeploymentModelInfo(deployment=raw_deployment, limits=limits),  # pyright: ignore[reportArgumentType]  # the model_list entry, as the core keeps it
        )
        self._invalidate_model_group_info_cache()

    def _invalidate_model_group_info_cache(self) -> None:
        self._ptu_cache = None
        self._cached_get_model_group_info.cache_clear()  # pyright: ignore[reportFunctionMemberAccess]  # an lru_cache
        self._core_cached_deployment_model_info.cache_clear()  # pyright: ignore[reportFunctionMemberAccess]  # an lru_cache
        self._zero_cost_cache.clear()
        self._routing_group_rows = None

    def set_routing_groups(self, groups: Sequence[RoutingGroup]) -> None:
        """``_init_routing_groups``' registry side: the groups by name, and the caches they feed."""
        self._core_routing_groups = {group.group_name: group for group in groups}
        self._invalidate_model_group_info_cache()
        self._invalidate_access_groups_cache()

    def _invalidate_access_groups_cache(self) -> None:
        self._access_groups_cache = None
        self._registry_version += 1

    def set_model_list(self, model_list: Sequence[object]) -> None:
        """Replace every deployment (``set_model_list``)."""
        self._discovered_model_info_cache.flush_cache()
        original_model_list: Final[list[UntypedEntry]] = copy.deepcopy(list(model_list))  # pyright: ignore[reportAssignmentType]  # model_list entries  # mutable-ok: a deep copy the build consumes
        with self._deployment_index_lock:
            self._core_model_list = []
            self.model_id_to_deployment_index_map = {}
            self._core_model_name_to_deployment_indices = {}
            self._core_team_model_to_deployment_indices = {}
        self._core_team_pattern_routers = {}
        self._core_team_public_model_names = frozenset()
        # The replaced list's wildcard routes and names go with it.
        self._core_pattern_router = PatternMatchRouter()
        self._core_provider_default_deployment_ids.clear()
        self._core_deployment_names.clear()
        if self.strategies is not None:
            self.strategies.reset()
        self._provider_unresolved_deployments = ()
        self._invalidate_model_group_info_cache()
        self._invalidate_access_groups_cache()
        declared_ids: Final = tuple(
            str(info["id"]) for entry in original_model_list if (info := _model_info_of(entry)).get("id") is not None
        )
        duplicate_ids: Final = frozenset(model_id for model_id in declared_ids if declared_ids.count(model_id) > 1)
        ptu_declared: Final = tuple(
            str(entry.get("model_name"))
            for entry in original_model_list
            if (ptu_info := _model_info_of(entry)).get("db_model") is not True and declares_ptu(ptu_info)
        )
        if ptu_declared and not is_ptu_cost_attribution_enabled():
            verbose_router_logger.warning(
                "PTU fields are set on config.yaml deployment(s) %s, but PTU cost attribution is disabled, so no "
                "flat cost accrues and this traffic is billed per token. Set %s=True to enable it",
                ", ".join(ptu_declared),
                PTU_COST_ATTRIBUTION_ENV_VAR,
            )
        for model in original_model_list:
            model_name = model.pop("model_name")
            raw_params = model.pop("litellm_params")
            litellm_params: UntypedEntry = raw_params if isinstance(raw_params, dict) else {}  # pyright: ignore[reportAssignmentType]  # configured params
            for key, value in litellm_params.items():
                if isinstance(value, str) and value.startswith("os.environ/") and not holds_secret_pointer(key):
                    litellm_params[key] = get_secret(value)
            model_info = model.pop("model_info", {})
            info_dict: UntypedEntry = model_info if isinstance(model_info, dict) else {}  # pyright: ignore[reportAssignmentType]  # a configured model_info
            declared_id = None if info_dict.get("id") is None else str(info_dict["id"])
            if "id" not in info_dict:
                info_dict["id"] = deployment_id(str(model_name), litellm_params)
            organization = litellm_params.get("organization")
            if isinstance(organization, list):
                # Each organization's deployment has its own id (``organization_deployment_id``).
                for position, org in enumerate(tuple(organization)):  # pyright: ignore[reportUnknownArgumentType, reportUnknownVariableType]  # configured organizations
                    litellm_params["organization"] = org
                    self._create_deployment(
                        model,
                        str(model_name),
                        litellm_params,
                        {
                            **info_dict,
                            "id": organization_deployment_id(
                                declared_id,
                                position,
                                org,
                                lambda: deployment_id(str(model_name), litellm_params),
                            ),
                        },
                        declared_id=declared_id,
                        duplicate_ids=duplicate_ids,
                    )
            else:
                self._create_deployment(
                    model,
                    str(model_name),
                    litellm_params,
                    info_dict,
                    declared_id=declared_id,
                    duplicate_ids=duplicate_ids,
                )
        self._core_model_names = {str(_LITELLM_PARAMS_ADAPTER.validate_python(m).get("model_name")) for m in model_list}
        if self.strategies is not None:
            self.strategies.finalize()
        self._on_change()

    def _create_deployment(
        self,
        deployment_info: UntypedEntry,
        model_name: str,
        litellm_params: UntypedEntry,
        model_info: UntypedEntry,
        *,
        declared_id: str | None = None,
        duplicate_ids: frozenset[str] = frozenset(),
    ) -> Deployment | None:
        """Validate one deployment and add it (``_create_deployment``); None when it is left out."""
        try:
            config_sourced: Final = model_info.get("db_model") is not True
            identity_error: Final = (
                ptu_identity_error(
                    declared_id=declared_id,
                    taken=declared_id in duplicate_ids,
                    current_id=model_info.get("id"),  # pyright: ignore[reportArgumentType]  # the configured id
                    model_name=model_name,
                )
                if config_sourced and ptu_terms(model_info) is not None
                else None
            )
            ptu_error: Final = (
                (ptu_config_error(model_info, model_name=model_name) or identity_error) if config_sourced else None
            )
            if ptu_error is not None and (declares_ptu_shares(model_info) or is_ptu_cost_attribution_enabled()):
                raise ValueError(ptu_error)
            access_windows_error: Final = access_windows_config_error(model_info, model_name=model_name)
            if access_windows_error is not None:
                raise ValueError(access_windows_error)
            capacity_warning: Final = (
                ptu_capacity_warning(
                    model_name, MappingProxyType({"model_info": model_info, "litellm_params": litellm_params})
                )
                if is_ptu_cost_attribution_enabled()
                else None
            )
            if capacity_warning is not None:
                verbose_router_logger.warning(capacity_warning)
            zeroed_pricing: Final = zeroed_ptu_pricing(model_info, litellm_params) if config_sourced else None
            merged_params: Final[Mapping[str, object]] = (
                litellm_params if zeroed_pricing is None else MappingProxyType({**litellm_params, **zeroed_pricing})
            )
            params: Final[LiteLLM_Params] = validate_litellm_params_strictly(LiteLLM_Params, dict(**merged_params))
            declared_params: Final = _LITELLM_PARAMS_ADAPTER.validate_python(litellm_params)
            warn_on_provider_credential_mismatch(model_name=model_name, litellm_params=declared_params)
            warn_on_ignored_model_capability_flags(model_name=model_name, litellm_params=declared_params)
            warn_on_unenforced_endpoint_flag(model_name=model_name, model_info=model_info)
            deployment: Final = without_model_capability_flags(
                Deployment(**deployment_info, model_name=model_name, litellm_params=params, model_info=model_info)  # pyright: ignore[reportArgumentType]  # the entry's extra fields, as the core passes them
            )
            register_deployment_pricing(self._host, deployment)
            if model_info_is_active_for_environment(model_info=deployment.model_info) is not True:  # pyright: ignore[reportArgumentType]  # ModelInfo reads like a mapping
                verbose_router_logger.warning(
                    "Ignoring deployment %s as it is not active for environment %s",
                    deployment.model_name,
                    deployment.model_info["supported_environments"],
                )
                return None
            for pattern in deployment.litellm_params.get("tag_regex") or []:
                try:
                    re.compile(pattern)
                except re.error as exc:
                    raise ValueError(
                        f"Invalid regex in tag_regex for model '{deployment.model_name}': {pattern!r} — {exc}"
                    ) from exc
            added: Final = self._add_deployment(deployment=with_admin_api_key_secret_reference(deployment))
            self._add_model_to_list_and_index_map(model=added.to_json(exclude_none=True), model_id=added.model_info.id)
        except Exception as e:
            if not self.ignore_invalid_deployments:
                raise
            if isinstance(e, litellm_exceptions.BadRequestError):
                retry: Final = partial(
                    self._create_deployment,
                    deployment_info,
                    model_name,
                    litellm_params,
                    model_info,
                    declared_id=declared_id,
                    duplicate_ids=duplicate_ids,
                )
                self._provider_unresolved_deployments = (*self._provider_unresolved_deployments, retry)
            verbose_router_logger.exception(
                "Error creating deployment: %s, ignoring and continuing with other deployments.", e
            )
            return None
        else:
            return added

    @staticmethod
    def _model_group_llm_provider(litellm_params: LiteLLM_Params) -> tuple[str, str]:
        try:
            return resolve_model_and_provider_for_metadata(
                model=litellm_params.model, custom_llm_provider=litellm_params.custom_llm_provider
            )
        except litellm_exceptions.BadRequestError as e:
            verbose_router_logger.error("litellm.router.py::get_model_group_info() - %s", e)
            return "", ""

    def _add_deployment(self, deployment: Deployment) -> Deployment:
        """Validate the provider and register the deployment's names and patterns (``_add_deployment``)."""
        litellm_model: Final = deployment.litellm_params.model
        if isinstance(deployment.litellm_params.drop_params, str):
            verbose_router_logger.warning(
                "model=%s drop_params=%r is not a flag value, treating it as unset",
                deployment.model_name,
                deployment.litellm_params.drop_params,
            )
        dynamic: Final = self._host.dynamic()
        prompt_management: Final = (
            "/" in litellm_model and litellm_model.split("/")[0] in dynamic.known_custom_logger_compatible_callbacks
        )
        custom_llm_provider: str | None
        if prompt_management:
            custom_llm_provider = None  # rebind-ok: one branch binds it
        elif declared_authenticating_provider(litellm_model, deployment.litellm_params.custom_llm_provider) is not None:
            _, custom_llm_provider = self._model_group_llm_provider(deployment.litellm_params)  # rebind-ok: see above
        else:
            _, custom_llm_provider, _, _ = self._host.invoke("get_llm_provider")(  # rebind-ok: see above
                model=deployment.litellm_params.model,
                custom_llm_provider=deployment.litellm_params.get("custom_llm_provider", None),
                api_base=deployment.litellm_params.api_base,
                litellm_params=deployment.litellm_params,
            )
            if (
                custom_llm_provider not in dynamic.provider_list
                and not JSONProviderRegistry.exists(custom_llm_provider)
                and not is_registered_custom_provider(custom_llm_provider)
            ):
                raise Exception(f"Unsupported provider - {custom_llm_provider}")  # noqa: TRY002  # the core raises a plain Exception
        self._core_deployment_names.append(deployment.litellm_params.model)
        if deployment.litellm_params.rpm is None and getattr(deployment, "rpm", None) is not None:
            deployment.litellm_params.rpm = getattr(deployment, "rpm")  # noqa: B009  # an extra field of the entry  # rebind-ok: the core sets it on the deployment it adds
        if deployment.litellm_params.tpm is None and getattr(deployment, "tpm", None) is not None:
            deployment.litellm_params.tpm = getattr(deployment, "tpm")  # noqa: B009  # an extra field of the entry  # rebind-ok: see above
        if "*" in deployment.model_name:
            self._core_pattern_router.add_pattern(deployment.model_name, deployment.to_json(exclude_none=True))
            if deployment.model_info.id:
                self._core_provider_default_deployment_ids.append(deployment.model_info.id)
        team_id: Final = deployment.model_info.get("team_id")
        team_public_model_name: Final[str | None] = deployment.model_info.get("team_public_model_name")
        if team_id is not None and team_public_model_name is not None and "*" in team_public_model_name:
            self._core_team_pattern_routers.setdefault(str(team_id), PatternMatchRouter()).add_pattern(
                team_public_model_name, deployment.to_json(exclude_none=True)
            )
        for data_source in deployment.litellm_params.get("dataSources", []) or []:
            source_params = data_source.get("parameters", {})
            for param_key in ("endpoint", "key"):
                if param_key in source_params and source_params[param_key].startswith("os.environ/"):
                    import os

                    source_params[param_key] = os.environ.get(source_params[param_key].replace("os.environ/", ""), "")
        if custom_llm_provider is not None:
            self._initialize_deployment_for_pass_through(deployment, custom_llm_provider)
        if self.strategies is not None:
            self.strategies.register(deployment)
        return deployment

    def _initialize_deployment_for_pass_through(self, deployment: Deployment, custom_llm_provider: str) -> None:
        """Register vertex credentials for a pass-through deployment (``_initialize_deployment_for_pass_through``)."""
        if deployment.litellm_params.use_in_pass_through is not True or custom_llm_provider != "vertex_ai":
            return
        credential_name: Final = deployment.litellm_params.litellm_credential_name
        credential_values: Final = (
            CredentialAccessor.get_credential_values(credential_name) if credential_name is not None else {}
        )
        project: Final = credential_values.get("vertex_project") or deployment.litellm_params.vertex_project
        location: Final = credential_values.get("vertex_location") or deployment.litellm_params.vertex_location
        credentials: Final = credential_values.get("vertex_credentials") or deployment.litellm_params.vertex_credentials
        if project is None or location is None:
            raise ValueError(
                "vertex_project, and vertex_location must be set in litellm_params for pass-through endpoints."
            )
        from litellm.router_host import router_host

        router_host().add_vertex_pass_through_credentials(
            project_id=str(project),
            location=str(location),
            vertex_credentials=credentials,  # pyright: ignore[reportArgumentType]  # the configured credentials
        )

    def served_deployments(self) -> list[UntypedEntry]:  # mutable-ok: a fresh list for the Rust handle
        """The deployments the Rust handle routes over: every entry with an id. Only
        ``router.model_list = [...]`` keeps an entry without one, and that entry is never served."""
        return [entry for entry in self._core_model_list if _model_info_of(entry).get("id") is not None]

    def assign_model_list(self, model_list: Sequence[object]) -> None:
        """``router.model_list = [...]`` (deprecated, ``_assign_model_list``): the list is swapped as
        given and reindexed, and the registry version moves once."""
        entries: Final = [_LITELLM_PARAMS_ADAPTER.validate_python(entry) for entry in model_list]
        with self._deployment_index_lock:
            self._core_model_list = model_list  # pyright: ignore[reportAttributeAccessIssue]  # the list as assigned, as the core keeps it
            self.model_id_to_deployment_index_map = {
                str(model_id): idx
                for idx, entry in enumerate(entries)
                if (model_id := _model_info_of(entry).get("id")) is not None
            }
            self._core_model_name_to_deployment_indices = {}
            self._core_team_model_to_deployment_indices = {}
            self._core_team_public_model_names = frozenset()
            # An entry without an id is kept as given but never served, as on the Python core.
            served: Final = [
                (idx, entry) for idx, entry in enumerate(entries) if _model_info_of(entry).get("id") is not None
            ]
            for idx, entry in served:
                model_name = entry.get("model_name")
                if model_name:
                    self._core_model_name_to_deployment_indices.setdefault(str(model_name), []).append(idx)
                self._update_team_model_index(entry, idx)
            self._core_model_names = {str(entry["model_name"]) for _, entry in served if entry.get("model_name")}
            self._access_groups_cache = None
            self._invalidate_model_group_info_cache()
            self._registry_version += 1
        self._on_change()

    def _update_team_model_index(self, model: Mapping[str, object], idx: int) -> None:
        info: Final = _model_info_of(model)
        team_id: Final = info.get("team_id")
        public_name: Final = info.get("team_public_model_name")
        if team_id and public_name:
            key: Final = (str(team_id), str(public_name))
            self._core_team_public_model_names = self._core_team_public_model_names | frozenset({str(public_name)})
            indices: Final = self._core_team_model_to_deployment_indices.setdefault(key, [])
            if idx not in indices:
                indices.append(idx)

    def _add_model_to_list_and_index_map(self, model: UntypedEntry, model_id: str | None = None) -> None:
        with self._deployment_index_lock:
            idx: Final = len(self._core_model_list)
            self._core_model_list.append(model)
            self._invalidate_model_group_info_cache()
            self._invalidate_access_groups_cache()
            info: Final = _model_info_of(model)
            if model_id is not None:
                self.model_id_to_deployment_index_map[model_id] = idx
            elif info.get("id") is not None:
                self.model_id_to_deployment_index_map[str(info["id"])] = idx
            model_name: Final = model.get("model_name")
            if model_name:
                self._core_model_name_to_deployment_indices.setdefault(str(model_name), []).append(idx)
            self._update_team_model_index(model, idx)

    def _update_deployment_indices_after_removal(self, model_id: str, removal_idx: int) -> None:
        self.model_id_to_deployment_index_map = {
            deployment: idx - 1 if idx > removal_idx else idx
            for deployment, idx in self.model_id_to_deployment_index_map.items()
            if deployment != model_id
        }
        for model_name, indices in list(self._core_model_name_to_deployment_indices.items()):
            kept = _without_indices(indices, removal_idx)
            if kept:
                self._core_model_name_to_deployment_indices[model_name] = kept
            else:
                del self._core_model_name_to_deployment_indices[model_name]
                self._core_model_names.discard(model_name)
        for key, indices in list(self._core_team_model_to_deployment_indices.items()):
            kept = _without_indices(indices, removal_idx)
            if kept:
                self._core_team_model_to_deployment_indices[key] = kept
            else:
                del self._core_team_model_to_deployment_indices[key]
        self._core_team_public_model_names = frozenset(
            public_model_name for _, public_model_name in self._core_team_model_to_deployment_indices
        )
        self._core_pattern_router.remove_deployment(model_id)
        if model_id in self._core_provider_default_deployment_ids:
            self._core_provider_default_deployment_ids.remove(model_id)
        for team_id in list(self._core_team_pattern_routers):
            team_router = self._core_team_pattern_routers[team_id]
            team_router.remove_deployment(model_id)
            if not team_router.patterns:
                del self._core_team_pattern_routers[team_id]

    # -- writes -----------------------------------------------------------------------------------

    def add_deployment(self, deployment: Deployment) -> Deployment | None:
        """Add one deployment unless its id exists (``add_deployment``)."""
        if deployment.model_info.id and self._core_has_model_id(deployment.model_info.id):
            return None
        declared_params: Final = deployment.litellm_params.model_dump(exclude_none=True)
        warn_on_provider_credential_mismatch(model_name=deployment.model_name, litellm_params=declared_params)
        warn_on_ignored_model_capability_flags(model_name=deployment.model_name, litellm_params=declared_params)
        warn_on_unenforced_endpoint_flag(
            model_name=deployment.model_name, model_info=deployment.model_info.model_dump()
        )
        served: Final = with_admin_api_key_secret_reference(without_model_capability_flags(deployment))
        entry: Final = served.to_json(exclude_none=True)
        self._add_deployment(deployment=served)
        register_deployment_pricing(self._host, served)
        self._add_model_to_list_and_index_map(model=entry, model_id=served.model_info.id)
        self._core_model_names.add(served.model_name)
        if self.checks is not None:
            self.checks.sync_deployment_budget(served)
        self._on_change()
        return served

    def _core_upsert_deployment(self, deployment: Deployment) -> Deployment | None:
        """Add or replace one deployment (``_upsert_deployment``)."""
        model_id: Final = deployment.model_info.id or ""
        on_router: Final = self._core_get_deployment(model_id=model_id)
        served: Final = without_model_capability_flags(deployment)
        try:
            if on_router is not None:
                if (
                    deployment.model_name == on_router.model_name
                    and served.litellm_params == on_router.litellm_params
                    and deployment.model_info == on_router.model_info
                ):
                    return None
                with self._deployment_index_lock:
                    removal_idx: Final = self.model_id_to_deployment_index_map.get(model_id)
                    if removal_idx is not None:
                        self._core_model_list.pop(removal_idx)
                        self._invalidate_model_group_info_cache()
                        self._invalidate_access_groups_cache()
                        self._update_deployment_indices_after_removal(model_id=model_id, removal_idx=removal_idx)
                if self.strategies is not None:
                    self.strategies.release(on_router)
                self._on_change()
            self.add_deployment(deployment=deployment)
            if self.strategies is not None:
                self.strategies.upserted(deployment, on_router)
        except Exception as e:
            if not self.ignore_invalid_deployments:
                # The failed edit raises, but the deployment it replaced keeps serving.
                self._restore_deployment_after_failed_upsert(on_router, model_id)
                raise
            verbose_router_logger.warning(
                "Error upserting deployment %s (id=%s): %s. Dropping it and continuing with other deployments.",
                deployment.model_name,
                deployment.model_info.id,
                e,
            )
            self._restore_deployment_after_failed_upsert(on_router, model_id)
            return None
        else:
            return deployment

    def _restore_deployment_after_failed_upsert(self, previous: Deployment | None, model_id: str) -> None:
        if previous is None or self._core_has_model_id(model_id):
            return
        try:
            if self.strategies is None:
                self.add_deployment(deployment=previous)
            else:
                self.strategies.without_capability_limit(lambda: self.add_deployment(deployment=previous))
            verbose_router_logger.info(
                "Restored deployment %s (id=%s); it keeps serving its previous configuration.",
                previous.model_name,
                model_id,
            )
        except Exception as restore_error:  # noqa: BLE001  # best-effort restore: a second failure must not abort the reload
            verbose_router_logger.warning(
                "Could not restore previously served deployment %s (id=%s) after the failed upsert: %s",
                previous.model_name,
                model_id,
                restore_error,
            )

    def _core_delete_deployment(self, id: str) -> UntypedEntry | None:
        """Remove one deployment and return its stored entry (``_delete_deployment``)."""
        try:
            with self._deployment_index_lock:
                deployment_idx: Final = self.model_id_to_deployment_index_map.get(id)
                if deployment_idx is None:
                    return None
                item: Final = self._core_model_list.pop(deployment_idx)
                self._invalidate_model_group_info_cache()
                self._invalidate_access_groups_cache()
                self._update_deployment_indices_after_removal(model_id=id, removal_idx=deployment_idx)
            if self.checks is not None:
                self.checks.unregister_deployment_budget(id)
            if self.strategies is not None:
                self.strategies.release_entry(item)
        except Exception:  # noqa: BLE001  # the core answers None for any failed delete
            return None
        self._on_change()
        return item

    def _core_get_router_deployment_budget_limiter(self) -> "RouterBudgetLimiting | None":
        """The router's deployment-budget callback (``registry().plugins("budget_limiter")``)."""
        return self.checks.budget_limiter() if self.checks is not None else None

    @property
    def _core_prompt_caching_pin_cache(self) -> DualCache:
        """The prompt caching pins (``registry().plugins("prompt_caching_pins")``)."""
        if self.checks is None:
            return self.__getattr__("_core_prompt_caching_pin_cache")  # pyright: ignore[reportReturnType]  # raises RustCoreUnsupported
        return self.checks.prompt_caching_pin_cache

    def replay_model_cost_registrations(self) -> None:
        """Re-register every served deployment's cost-map entries (``replay_model_cost_registrations``)."""
        provider_unresolved: Final = self._provider_unresolved_deployments
        self._provider_unresolved_deployments = ()
        for create_deployment in provider_unresolved:
            create_deployment()
        for entry in tuple(self._core_model_list):
            try:
                deployment = Deployment(**entry)  # pyright: ignore[reportArgumentType]  # a stored entry
            except Exception:  # noqa: BLE001  # a malformed entry must not abort the rest of the rebuild
                verbose_router_logger.exception(
                    "Router: could not rebuild cost-map entry for a deployment during a price data reload"
                )
                continue
            register_deployment_pricing(self._host, deployment)
        self._invalidate_model_group_info_cache()
        if provider_unresolved:
            self._on_change()

    # -- one deployment ---------------------------------------------------------------------------

    def _core_has_model_id(self, candidate_id: str) -> bool:
        return candidate_id in self.model_id_to_deployment_index_map

    def _core_get_model_info(self, id: str) -> UntypedEntry | None:
        with self._deployment_index_lock:
            idx: Final = self.model_id_to_deployment_index_map.get(id)
            return None if idx is None else self._core_model_list[idx]

    def _core_get_deployment(self, model_id: str) -> Deployment | None:
        entry: Final = self._core_get_model_info(model_id)
        return None if entry is None else Deployment(**entry)  # pyright: ignore[reportArgumentType]  # a stored entry

    def _core_get_deployment_by_model_group_name(self, model_group_name: str) -> Deployment | None:
        with self._deployment_index_lock:
            indices: Final = self._core_model_name_to_deployment_indices.get(model_group_name)
            model: Final = self._core_model_list[indices[0]] if indices else None
        return None if model is None else Deployment(**model)  # pyright: ignore[reportArgumentType]  # a stored entry

    def _core_deployments_named(self, model_name: str) -> tuple[UntypedEntry, ...] | None:
        with self._deployment_index_lock:
            indices: Final = self._core_model_name_to_deployment_indices.get(model_name)
            return None if indices is None else tuple(self._core_model_list[idx] for idx in indices)

    def team_public_model_names_of(self, team_id: str) -> frozenset[str]:
        with self._deployment_index_lock:
            return frozenset(name for owner, name in self._core_team_model_to_deployment_indices if owner == team_id)

    def _team_deployments_named(self, team_id: str, public_model_name: str) -> tuple[UntypedEntry, ...] | None:
        with self._deployment_index_lock:
            indices: Final = self._core_team_model_to_deployment_indices.get((team_id, public_model_name))
            return None if indices is None else tuple(self._core_model_list[idx] for idx in indices)

    # -- deployments by name ----------------------------------------------------------------------

    @staticmethod
    def _core_should_include_deployment(
        model_name: str, model: Mapping[str, object], team_id: str | None = None
    ) -> bool:
        info: Final = _model_info_of(model)
        if team_id is not None and info.get("team_id") == team_id and model_name == info.get("team_public_model_name"):
            return True
        if model_name is not None and model.get("model_name") == model_name:  # pyright: ignore[reportUnnecessaryComparison]  # the core checks it
            model_team_id: Final = info.get("team_id")
            return team_id is None or model_team_id is None or model_team_id == team_id
        return False

    @staticmethod
    def _core_deployment_usable_by_team(model: Mapping[str, object] | Deployment, team_id: str | None) -> bool:
        info: Final[object] = (
            model.model_info.model_dump() if isinstance(model, Deployment) else model.get("model_info")
        )
        owner: Final = _MODEL_INFO_ADAPTER.validate_python(info).get("team_id") if isinstance(info, Mapping) else None
        return team_may_use_deployment(owner, team_id)

    def _core_deployments_for_request(
        self, model: str, request_kwargs: Mapping[str, object]
    ) -> Sequence[Mapping[str, object]]:
        """The deployments ``model`` names for this caller: alias, then team-first, then global, then
        admin-across-teams (``_core_deployments_for_request``)."""
        from litellm.litellm_core_utils.routing_common_utils import get_request_team_id, is_proxy_admin_request

        registered: Final = self._core_get_model_from_alias(model=model) or model
        team_id: Final = get_request_team_id(request_kwargs)
        deployments: Final = self._core_get_all_deployments(model_name=registered, team_id=team_id)
        if deployments or team_id is not None or not is_proxy_admin_request(request_kwargs):
            return deployments
        return self._team_deployments_across_teams(registered)

    def _team_deployments_across_teams(self, model: str) -> list[UntypedEntry]:  # mutable-ok: a fresh list
        """Every team's deployments under public name ``model``, for a proxy admin calling without a team."""
        with self._deployment_index_lock:
            team_deployments: Final = [  # comprehension-ok: as the Python router collects them
                self._core_model_list[index]
                for (_, public_model_name), indices in self._core_team_model_to_deployment_indices.items()
                if public_model_name == model
                for index in indices
            ]
        team_ids: Final = {
            team_id for team_id in (_model_info_of(d).get("team_id") for d in team_deployments) if team_id is not None
        }
        if len(team_ids) > 1:
            raise litellm_exceptions.BadRequestError(
                message=(
                    f"Model name '{model}' matches deployments from multiple teams. "
                    "Specify the deployment ID directly to disambiguate."
                ),
                model=model,
                llm_provider="",
            )
        return team_deployments

    def _core_get_all_deployments(
        self, model_name: str, model_alias: str | None = None, team_id: str | None = None
    ) -> list[UntypedEntry]:  # mutable-ok: a fresh list, as the core returns it
        def named(model: UntypedEntry) -> UntypedEntry:
            return model if model_alias is None else _renamed(model, model_alias)

        if team_id is not None:
            team_models: Final = self._team_deployments_named(team_id, model_name)
            if team_models is not None:
                found: Final = [
                    named(model)
                    for model in team_models
                    if self._core_should_include_deployment(model_name=model_name, model=model, team_id=team_id)
                ]
                if found:
                    return found
        named_models: Final = self._core_deployments_named(model_name)
        if named_models is not None:
            return [
                named(model)
                for model in named_models
                if self._core_should_include_deployment(model_name=model_name, model=model, team_id=team_id)
            ]
        if team_id is not None:
            return [
                named(model)
                for model in self._core_model_list
                if self._core_should_include_deployment(model_name=model_name, model=model, team_id=team_id)
            ]
        return []

    def get_routing_group(self, model_name: str) -> RoutingGroup | None:
        """The routing group callable as ``model_name``: a deployment model_name or an alias of the
        same name shadows it."""
        if not self._core_routing_groups:
            return None
        group: Final = self._core_routing_groups.get(model_name)
        if (
            group is None
            or model_name in self._core_model_name_to_deployment_indices
            or model_name in (self._core_model_group_alias or {})
        ):
            return None
        return group

    def _core_routing_group_has_alternatives(self, model_group: str | None) -> bool:
        """A callable routing group whose member union spans more than one deployment."""
        if model_group is None:
            return False
        resolved: Final = self._core_get_model_from_alias(model=model_group) or model_group
        group: Final = self.get_routing_group(resolved)
        if group is None:
            return False
        return sum(len(self._core_model_name_to_deployment_indices.get(member) or ()) for member in group.models) > 1

    def _core_get_model_list_from_model_alias(
        self, model_name: str | None = None
    ) -> list[UntypedEntry]:  # mutable-ok: a fresh list
        if model_name is not None and model_name not in self._core_model_group_alias:
            return []
        alias_items: Final[Sequence[tuple[str, object]]] = (
            [(model_name, self._core_model_group_alias[model_name])]
            if model_name is not None
            else list(self._core_model_group_alias.items())
        )
        returned: Final[list[UntypedEntry]] = []  # mutable-ok: filled per alias
        for model_alias, model_value in alias_items:
            if isinstance(model_value, str):
                target = model_value
            elif isinstance(model_value, dict):
                item = RouterModelGroupAliasItem(**model_value)  # pyright: ignore[reportCallIssue, reportArgumentType]  # an alias item dict
                if item["hidden"] is True and model_name is None:
                    continue
                target = item["model"]
            else:
                continue
            alias_group = self.get_routing_group(target)
            if alias_group is not None:
                returned.extend(
                    {**row, "model_name": model_alias} for row in self._materialize_routing_group_rows((alias_group,))
                )
            else:
                returned.extend(self._core_get_all_deployments(model_name=target, model_alias=model_alias))
        return returned

    def _core_get_model_list_from_routing_groups(self, model_name: str | None = None) -> Sequence[UntypedEntry]:
        """Callable routing groups as model-list rows: each member deployment under the group's name."""
        if model_name is not None:
            group: Final = self.get_routing_group(model_name)
            return self._materialize_routing_group_rows((group,)) if group is not None else ()
        cached: Final = self._routing_group_rows
        if cached is not None:
            return cached
        rows: Final = self._materialize_routing_group_rows(
            tuple(
                callable_group
                for name in self._core_routing_groups
                if (callable_group := self.get_routing_group(name)) is not None
            )
        )
        self._routing_group_rows = rows
        return rows

    def _materialize_routing_group_rows(self, groups: tuple[RoutingGroup, ...]) -> tuple[UntypedEntry, ...]:
        return tuple(  # comprehension-ok: each member deployment of each group, in order
            _as_routing_group_row(apply_routing_group_priority(group, member, deployment))  # pyright: ignore[reportArgumentType]  # a stored entry
            for group in groups
            for member in group.models
            for deployment in self._core_get_all_deployments(model_name=member, model_alias=group.group_name)
        )

    def _get_wildcard_deployments(
        self, model_name: str, team_id: str | None = None
    ) -> list[UntypedEntry]:  # mutable-ok: a fresh list
        team_router: Final = self._core_team_pattern_routers.get(team_id) if team_id is not None else None
        matches: Final = [
            *self._core_pattern_router.get_deployments_by_pattern(model=model_name),
            *(team_router.get_deployments_by_pattern(model=model_name) if team_router is not None else ()),
        ]
        return [{**DeploymentTypedDict(**m), "model_name": model_name} for m in matches]  # pyright: ignore[reportCallIssue]  # a stored entry

    def _core_get_model_list(
        self, model_name: str | None = None, team_id: str | None = None
    ) -> list[UntypedEntry] | None:  # mutable-ok: a fresh list, as get_model_list returns
        returned: Final[list[UntypedEntry]] = []  # mutable-ok: built in the core's order
        if model_name is not None:
            returned.extend(self._core_get_all_deployments(model_name=model_name, team_id=team_id))
        returned.extend(self._core_get_model_list_from_model_alias(model_name=model_name))
        returned.extend(self._core_get_model_list_from_routing_groups(model_name=model_name))
        if not returned and model_name is not None:
            returned.extend(self._get_wildcard_deployments(model_name=model_name, team_id=team_id))
        if model_name is None:
            returned.extend(self._core_model_list)
        return returned

    def _core_wildcard_route_deployments(self, model: str, team_id: str | None) -> Sequence[Mapping[str, object]]:
        team_router: Final = self._core_team_pattern_routers.get(team_id) if team_id is not None else None
        team_owned: Final[Sequence[Mapping[str, object]]] = (
            team_router.get_deployments_by_pattern(model=model) if team_router is not None else ()
        )
        return team_owned or self._core_pattern_router.get_deployments_by_pattern(model=model)

    def _core_wildcard_routed_models(self, model: str, team_id: str | None) -> frozenset[str]:
        team_router: Final = self._core_team_pattern_routers.get(team_id) if team_id is not None else None
        team_owned: Final = (
            team_router.get_models_by_pattern(model=model) if team_router is not None else frozenset[str]()
        )
        return team_owned or self._core_pattern_router.get_models_by_pattern(model=model)

    def _core_config_deployments(self) -> Iterable[Mapping[str, object]]:
        return (entry for entry in self._core_model_list if _model_info_of(entry).get("db_model") is not True)

    # -- names ------------------------------------------------------------------------------------

    def _core_get_model_names(self, team_id: str | None = None) -> list[str]:  # mutable-ok: a fresh list
        names: Final[list[str]] = []  # mutable-ok: filled per deployment
        for deployment in self._core_get_model_list() or []:
            info = _model_info_of(deployment)
            if info.get("team_id"):
                if team_id == info.get("team_id"):
                    public = info.get("team_public_model_name")
                    if public:
                        names.append(str(public))
            else:
                names.append(str(deployment.get("model_name", "")))
        return names

    def _core_get_model_ids(
        self, model_name: str | None = None, exclude_team_models: bool = False
    ) -> list[str]:  # mutable-ok: a fresh list
        if model_name is not None:
            return [
                str(info["id"])
                for model in self._core_deployments_named(model_name) or ()
                if "id" in (info := _model_info_of(model)) and not (exclude_team_models and info.get("team_id"))
            ]
        with self._deployment_index_lock:
            id_indices: Final = tuple(self.model_id_to_deployment_index_map.items())
            models: Final = tuple(self._core_model_list)
        return [
            model_id
            for model_id, idx in id_indices
            if "id" in (each_info := _model_info_of(models[idx]))
            and not (exclude_team_models and each_info.get("team_id"))
        ]

    # -- resolving a name -------------------------------------------------------------------------

    def _core_get_model_from_alias(self, model: str) -> str | None:
        return resolve_model_group_alias(self._core_model_group_alias, model)

    def _core_routable_model_group(self, model_name: str) -> str:
        target: Final = self._core_get_model_from_alias(model_name)
        return target if target is not None else model_name

    def _core_is_recognized_model(self, model: str) -> bool:
        return (
            model in self._core_model_names
            or self._core_has_model_id(model)
            or model in self._core_model_group_alias
            or self.get_routing_group(model) is not None
        )

    def _core_get_model_group(self, id: str) -> list[UntypedEntry] | None:  # mutable-ok: a fresh list
        model_info: Final = self._core_get_model_info(id=id)
        if model_info is None:
            return None
        return self._core_get_model_list(model_name=str(model_info["model_name"]))

    def _core_resolve_model_name_from_model_id(
        self, model_id: str | None, custom_llm_provider: str | None = None
    ) -> str | None:
        if not model_id:
            return None
        if model_id in self._core_model_names:
            return model_id
        if self._core_has_model_id(model_id):
            deployment: Final = self._core_get_deployment(model_id=model_id)
            if deployment is not None and deployment.model_name:
                return deployment.model_name
            return model_id
        all_models: Final = self._core_get_model_list(model_name=None)
        if not all_models:
            return None
        for entry in all_models:
            actual_model = _params_of(entry).get("model")
            if isinstance(actual_model, str) and (
                actual_model == model_id or actual_model.endswith((f"/{model_id}", f":{model_id}"))
            ):
                model_name = entry.get("model_name")
                if model_name:
                    return str(model_name)
        if self._core_is_recognized_model(model_id):
            return model_id
        if custom_llm_provider is None:
            return None
        if self._core_pattern_router.routes_to_provider(model_id, custom_llm_provider):
            return model_id
        provider_prefixed: Final = f"{custom_llm_provider}/{model_id}"
        if not model_id.startswith(f"{custom_llm_provider}/") and self._core_pattern_router.routes_to_provider(
            provider_prefixed, custom_llm_provider
        ):
            return provider_prefixed
        return None

    def _get_deployment_by_litellm_model(self, model: str) -> list[UntypedEntry]:  # mutable-ok: a fresh list
        return [m for m in self._core_model_list if _params_of(m).get("model") == model]

    # -- credentials ------------------------------------------------------------------------------

    @staticmethod
    def _core_is_deployment_blocked(deployment: Deployment) -> bool:
        model_info: Final[object] = getattr(deployment, "model_info", None)
        return model_info is not None and getattr(model_info, "blocked", None) is True

    def _core_get_deployment_credentials(self, model_id: str) -> UntypedEntry | None:
        deployment: Final = self._core_get_deployment(model_id=model_id)
        if deployment is None or self._core_is_deployment_blocked(deployment):
            return None
        return CredentialLiteLLMParams.model_validate(
            deployment.litellm_params.model_dump(exclude_none=True)
        ).model_dump(exclude_none=True)

    def _get_model_group_deployment_usable_by_team(
        self, model_group_name: str, team_id: str | None
    ) -> Deployment | None:
        with self._deployment_index_lock:
            indices: Final = self._core_model_name_to_deployment_indices.get(model_group_name) or ()
            first_usable: Final = next(
                (
                    self._core_model_list[idx]
                    for idx in indices
                    if self._core_deployment_usable_by_team(self._core_model_list[idx], team_id)
                ),
                None,
            )
        return None if first_usable is None else Deployment(**first_usable)  # pyright: ignore[reportArgumentType]  # a stored entry

    def _get_team_public_name_deployment(self, model_id: str, team_id: str | None) -> Deployment | None:
        if team_id is None:
            return None
        with self._deployment_index_lock:
            team_indices: Final = self._core_team_model_to_deployment_indices.get((team_id, model_id))
            team_model: Final = self._core_model_list[team_indices[0]] if team_indices else None
        return None if team_model is None else Deployment(**team_model)  # pyright: ignore[reportArgumentType]  # a stored entry

    def _get_wildcard_deployment_usable_by_team(self, model_id: str, team_id: str | None) -> Deployment | None:
        team_router: Final = self._core_team_pattern_routers.get(team_id) if team_id is not None else None
        team_models: Final = team_router.route(model_id) if team_router else None
        global_models: Final = tuple(
            model
            for model in (self._core_pattern_router.route(model_id) or ())
            if self._core_deployment_usable_by_team(model, team_id)
        )
        candidates: Final = team_models or global_models
        if not candidates:
            return None
        return Deployment(**candidates[0])  # a pattern router holds stored entry dicts

    def _core_get_credential_deployment(self, model_id: str, team_id: str | None = None) -> Deployment | None:
        deployment: Final = (
            self._core_get_deployment(model_id=model_id)
            or self._get_model_group_deployment_usable_by_team(model_group_name=model_id, team_id=team_id)
            or self._get_team_public_name_deployment(model_id=model_id, team_id=team_id)
            or self._get_wildcard_deployment_usable_by_team(model_id=model_id, team_id=team_id)
        )
        if deployment is None or self._core_is_deployment_blocked(deployment):
            return None
        return deployment

    def _core_get_deployment_credentials_with_provider(
        self, model_id: str, team_id: str | None = None
    ) -> UntypedEntry | None:
        deployment: Final = self._core_get_credential_deployment(model_id=model_id, team_id=team_id)
        if deployment is None:
            return None
        base: Final = CredentialLiteLLMParams.model_validate(
            deployment.litellm_params.model_dump(exclude_none=True)
        ).model_dump(exclude_none=True)
        credential_name: Final = deployment.litellm_params.litellm_credential_name
        resolved: Final[UntypedEntry] = dict(base)
        if credential_name is not None:
            credential_values: Final = CredentialAccessor.get_credential_values(credential_name)
            if not credential_values:
                verbose_router_logger.warning("Credential '%s' not found in credential_list", credential_name)
            resolved.update(credential_values)
            _ = resolved.pop("litellm_credential_name", None)
        model: Final = deployment.litellm_params.model
        resolved["model"] = model
        resolved["custom_llm_provider"] = (
            deployment.litellm_params.custom_llm_provider or (model.split("/")[0] if "/" in model else None) or "openai"
        )
        return resolved

    # -- routable names ---------------------------------------------------------------------------

    def _routable_deployments(self, model_name: str, team_id: str | None) -> tuple[UntypedEntry, ...]:
        named: Final = self._core_get_all_deployments(
            model_name=self._core_routable_model_group(model_name), team_id=team_id
        )
        usable: Final = tuple(d for d in named if self._core_deployment_usable_by_team(d, team_id))
        return tuple(d for d in (usable or named) if _model_info_of(d).get("blocked") is not True)

    def _core_team_ptu_ceiling(self, team_id: str, model_group: str) -> "PTUTeamCeiling | None":
        """The per-minute ceiling ``team_id``'s PTU shares set on the group serving ``model_group``,
        else None (``team_ptu_ceiling``), kept until the model list next changes."""
        from litellm.litellm_core_utils.ptu_shares import ptu_share_teams, team_ptu_ceiling

        listed: Final = tuple(self._core_get_model_list() or ())
        if self._ptu_cache is None:
            self._ptu_cache = (ptu_share_teams(listed), {})
        share_teams, ceilings = self._ptu_cache
        if team_id not in share_teams:
            return None
        key: Final = (team_id, model_group)
        if key not in ceilings:
            ceilings[key] = team_ptu_ceiling(
                listed,
                self._core_model_list,
                team_id,
                model_group,
                lambda name: self._core_wildcard_route_deployments(name, team_id),
            )
        return ceilings[key]

    def _get_client(
        self, deployment: Mapping[str, object], kwargs: Mapping[str, object], client_type: str | None = None
    ) -> object:
        """The SDK client cached in the routing cache for a deployment (``_get_client``); a pod
        discovery deployment has none. ``max_parallel_requests`` is the deployment's slot limit,
        built on first use."""
        from litellm.litellm_core_utils.core_helpers import get_parent_otel_span_from_kwargs
        from litellm.router_rust_core import RustCoreUnsupported

        # `deployment["model_info"]["id"]`: the Python router reads the id first, raising for a
        # deployment without one whatever the client type.
        _ = deployment["model_info"]["id"]  # pyright: ignore[reportIndexIssue]  # read as the Python router reads it
        if client_type == "max_parallel_requests":
            limit_lookup: Final = self.max_parallel_requests_lookup
            if limit_lookup is None:
                raise RustCoreUnsupported(
                    "registry().client(client_type='max_parallel_requests') is not served by the Rust router core"
                )
            return limit_lookup(deployment, kwargs)
        if _params_of(deployment).get("kubernetes_pod_discovery") is True:
            return None
        model_id: Final = _model_info_of(deployment).get("id")
        span: Final = get_parent_otel_span_from_kwargs(dict(kwargs))
        stream: Final = "_stream" if kwargs.get("stream") is True else ""
        with service_target(ROUTER_CLIENTS_TARGET):
            if client_type == "async":
                return self._core_cache.get_cache(
                    key=f"{model_id}{stream}_async_client", local_only=True, parent_otel_span=span
                )  # pyright: ignore[reportUnknownMemberType]  # DualCache's reads are untyped
            return self._core_cache.get_cache(key=f"{model_id}{stream}_client", parent_otel_span=span)  # pyright: ignore[reportUnknownMemberType]  # see above

    def _core_get_search_tool_support(self, model_name: str, team_id: str | None = None) -> tuple[bool | None, ...]:
        """Per routable deployment: its ``supports_search_tool`` when a bool, else True when its route
        builds an Anthropic Messages request, else None (``get_search_tool_support``)."""
        return tuple(_search_tool_support(deployment) for deployment in self._routable_deployments(model_name, team_id))

    def _core_get_candidate_model_ids_for_route(self, model: str, team_id: str | None = None) -> frozenset[str]:
        """The deployment ids that could serve ``model`` for ``team_id``: the alias target, then a
        routing group, then a team public name or wildcard route for a name that is no model
        group, then the model group (``get_candidate_model_ids_for_route``). Read-only."""
        resolved: Final = self._core_get_model_from_alias(model=model) or model
        group: Final = self.get_routing_group(resolved) if self._core_routing_groups else None
        if group is not None:
            # The member union (`_get_routing_group_deployments`); a priority changes no id.
            return frozenset[str]().union(
                *(
                    _deployment_ids(self._core_get_all_deployments(model_name=member, team_id=team_id))
                    for member in group.models
                )
            )
        if resolved not in self._core_model_names:
            team_deployments: Final = (
                self._core_get_all_deployments(model_name=resolved, team_id=team_id) if team_id is not None else []
            )
            if team_deployments:
                return _deployment_ids(team_deployments)
            wildcard: Final = self._core_wildcard_route_deployments(resolved, team_id)
            if wildcard:
                return _deployment_ids(wildcard)
        return _deployment_ids(self._core_get_all_deployments(model_name=resolved, team_id=team_id))

    def _core_team_model_has_alternatives(self, deployment_id: str) -> bool:
        """Whether a team deployment's public name has another unpaused deployment of that team."""
        deployment: Final = self._core_get_deployment(model_id=deployment_id)
        if deployment is None:
            return False
        team_id: Final = deployment.model_info.team_id
        public_model_name: Final = deployment.model_info.team_public_model_name
        if team_id is None or public_model_name is None:
            return False
        siblings: Final = self._team_deployments_named(team_id, public_model_name) or ()
        return len([s for s in siblings if _model_info_of(s).get("blocked") is not True]) > 1

    def _core_resolve_served_model_name(self, model_name: str, served_model: str) -> str:
        """``served_model`` when ``model_name`` routes to it (the alias target, a routing group
        member, or the name itself), else ``model_name``."""
        target: Final = self._core_get_model_from_alias(model_name) or model_name
        group: Final = self.get_routing_group(target)
        members: Final = group.models if group is not None else ()
        return served_model if served_model == target or served_model in members else model_name

    def _core_resolved_litellm_models(
        self, model_name: str, team_id: str | None = None, *, use_base_model: bool = False
    ) -> tuple[str, ...]:
        """The provider model strings a call to ``model_name`` can be served by (``resolved_litellm_models``)."""
        listed: Final = self._core_get_model_list(model_name=model_name, team_id=team_id) or ()
        if use_base_model:
            return tuple(
                backend for entry in listed if (backend := cost_map_backend_model(Deployment.model_validate(entry)))
            )
        return tuple(model for entry in listed if isinstance(model := _params_of(entry).get("model"), str) and model)

    def _core_get_routable_deployments(
        self, model: str, request_kwargs: Mapping[str, object] = MappingProxyType({}), call_type: str | None = None
    ) -> tuple[Mapping[str, object], ...]:
        """The deployments routing would consider for ``request_kwargs``, as the Rust core resolves them
        (endpoint eligibility for ``call_type`` is not applied: the Rust resolution has none)."""
        _ = call_type
        lookup: Final = self.routable_lookup
        if lookup is None:
            from litellm.router_rust_core import RustCoreUnsupported

            raise RustCoreUnsupported("registry().get_routable_deployments is not served by the Rust router core")
        return tuple(lookup(model, request_kwargs))

    def _core_get_routable_upstream_model(self, model_name: str, team_id: str | None = None) -> str | None:
        deployment: Final = next(iter(self._routable_deployments(model_name, team_id)), None)
        if deployment is None:
            return None
        model: Final = _params_of(deployment).get("model")
        return None if model is None else str(model)

    def _core_map_team_model(
        self, team_model_name: str | None, team_id: str, custom_llm_provider: str | None = None
    ) -> str | None:
        """A team's public name to its group, or a deployment for a model-less call (``map_team_model``)."""
        models: Final = self._core_get_model_list(model_name=team_model_name, team_id=team_id)
        if not models:
            return None
        if team_model_name is None and custom_llm_provider is not None:
            return self._team_model_for_provider(models, team_id, custom_llm_provider)
        for model in models:
            info = _model_info_of(model)
            if info.get("team_id") == team_id:
                if team_model_name is None:
                    public = info.get("team_public_model_name") or model.get("model_name")
                    return None if public is None else str(public)
                return team_model_name
        return None

    @staticmethod
    def _team_model_for_provider(models: Sequence[UntypedEntry], team_id: str, custom_llm_provider: str) -> str | None:
        def public_name(deployment: UntypedEntry) -> str | None:
            info: Final = _model_info_of(deployment)
            if info.get("team_id") != team_id:
                return None
            name: Final = info.get("team_public_model_name")
            return name if isinstance(name, str) and name else str(deployment["model_name"])

        def provider(deployment: UntypedEntry) -> object:
            return provider_for_generic_call(_params_of(deployment))  # pyright: ignore[reportArgumentType]  # the deployment's params

        matching: Final = next(
            (m for m in models if public_name(m) is not None and provider(m) == custom_llm_provider), None
        )
        if matching is None:
            return None
        name: Final = public_name(matching)
        if name is None:
            return None
        if all(provider(m) == custom_llm_provider for m in models if public_name(m) == name):
            return name
        matched_id: Final = _model_info_of(matching).get("id")
        return matched_id if isinstance(matched_id, str) else None

    # -- access groups ----------------------------------------------------------------------------

    def _core_get_model_access_groups(
        self, model_name: str | None = None, model_access_group: str | None = None, team_id: str | None = None
    ) -> Mapping[str, Sequence[str]]:
        use_cache: Final = model_name is None and model_access_group is None and team_id is None
        if use_cache and self._access_groups_cache is not None:
            return self._access_groups_cache
        groups: Final[dict[str, list[str]]] = {}  # mutable-ok: built here, as the core's defaultdict
        for entry in self._core_get_model_list(model_name=model_name, team_id=team_id) or []:
            info = entry.get("model_info")
            if not info:
                continue
            for group in _model_info_of(entry).get("access_groups", []) or []:  # pyright: ignore[reportGeneralTypeIssues]  # a configured list
                if model_access_group is None or group == model_access_group:
                    groups.setdefault(str(group), []).append(str(entry["model_name"]))
        if use_cache:
            self._access_groups_cache = groups
        return groups

    def _core_is_model_access_group_for_wildcard_route(self, model_access_group: str) -> bool:
        groups: Final = self._core_get_model_access_groups(model_access_group=model_access_group)
        return any(
            self._core_pattern_router.route(request=model) is not None for model in groups.get(model_access_group, [])
        )


# -- the registry() view -------------------------------------------------------------------------


class RustRegistry(RouterRegistry):
    """``registry()`` of the Rust core: ``RouterRegistry``'s queries over the host registry."""

    __slots__ = ()

    def includes(
        self,
        deployment: "DeploymentTypedDict | dict[str, object]",  # mutable-ok: the RouterRegistry signature
        *,
        model_name: str | None = None,
        team_id: str | None = None,
    ) -> bool:
        hosted: Final = self._router
        if not isinstance(hosted, HostRegistry):
            return super().includes(deployment, model_name=model_name, team_id=team_id)
        if model_name is None:
            return HostRegistry._core_deployment_usable_by_team(deployment, team_id)  # pyright: ignore[reportPrivateUsage, reportArgumentType]  # the view reads its own registry
        return HostRegistry._core_should_include_deployment(model_name=model_name, model=deployment, team_id=team_id)  # pyright: ignore[reportPrivateUsage, reportArgumentType]  # see above


def hosted_registry_view(hosted: HostRegistry) -> RouterRegistry:
    """The ``registry()`` view over ``hosted``."""
    return RustRegistry(hosted)  # pyright: ignore[reportArgumentType]  # the host registry answers every _core_ query the view makes


# The registry indexes a deprecated attribute write replaces (``router.model_names = ...`` reaches
# the host registry as ``_core_model_names``), by the field router-core reads each one as.
INDEX_ATTRIBUTES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "_core_model_names": "model_names",
        "_core_model_name_to_deployment_indices": "model_name",
        "_core_team_model_to_deployment_indices": "team_model",
    }
)


def _positions(indices: object, length: int) -> list[int]:  # mutable-ok: a JSON list
    """The model-list positions ``indices`` read, as Python's ``model_list[index]`` reads them: a
    negative one counts from the end, and one before the start becomes ``length``, a position past
    the end, so the lookup raises ``IndexError`` as Python's does."""
    if not isinstance(indices, Iterable) or isinstance(indices, str | bytes):
        return []
    return [
        index if index >= 0 else (index + length if index + length >= 0 else length)
        for index in indices
        if isinstance(index, int) and not isinstance(index, bool)
    ]


def index_document(hosted: HostRegistry, written: Iterable[str]) -> Mapping[str, object]:
    """The written registry indexes, as the host registry holds them now, in the shape router-core
    reads (``IndexOverrides``): names a lookup finds, and the deployment positions each model name
    (or ``(team_id, public name)`` pair) indexes. Entries Python could not read as such are left out."""
    document: Final[dict[str, object]] = {}  # mutable-ok: built here
    length: Final = len(hosted._core_model_list)  # pyright: ignore[reportPrivateUsage]  # positions read against the live list
    for attribute in written:
        value: object = getattr(hosted, attribute, None)
        field: str = INDEX_ATTRIBUTES[attribute]
        if field == "model_names":
            names = value if isinstance(value, Iterable) and not isinstance(value, str | bytes | Mapping) else ()
            document[field] = sorted({name for name in names if isinstance(name, str)})
        elif field == "model_name":
            entries = value if isinstance(value, Mapping) else {}
            document[field] = {
                name: _positions(indices, length) for name, indices in entries.items() if isinstance(name, str)
            }
        else:
            pairs = value if isinstance(value, Mapping) else {}
            document[field] = [
                [key[0], key[1], _positions(indices, length)]
                for key, indices in pairs.items()
                if isinstance(key, tuple) and len(key) == 2 and all(isinstance(part, str) for part in key)
            ]
    return document


DEFAULT_DEPLOYMENT_FAILING_ID: Final = "__litellm_default_deployment_failing_call__"
"""The id router-core serves a ``default_deployment`` under when every call to it fails."""

DEFAULT_DEPLOYMENT_COPIED_ID: Final = "__litellm_default_deployment_copied__"
"""The id router-core serves a callable ``default_deployment`` under when its ``model_info.id`` is
not a string: the host picks and calls Python's copy of it, so its id stays what was assigned."""


def default_deployment_copy(value: object, model: str) -> object:
    """``_try_early_resolve_deployments_for_model_not_in_names``' copy of the default deployment
    for ``model``, raising what it raises for a value it cannot copy."""
    updated: Final = value.copy()  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownVariableType]  # any assigned value, read as Python reads it
    updated["litellm_params"] = value["litellm_params"].copy()  # pyright: ignore[reportIndexIssue, reportUnknownMemberType]  # same
    updated["litellm_params"]["model"] = model  # pyright: ignore[reportUnknownMemberType]  # same
    return updated  # pyright: ignore[reportUnknownVariableType]  # same


def default_deployment_copy_error(value: object, model: str) -> BaseException:
    """The error the default deployment's copy raises for ``model``."""
    try:
        _ = default_deployment_copy(value, model)
    except Exception as raised:  # noqa: BLE001  # the error the pick raises
        return raised
    return RuntimeError(f"router-core: the default deployment copies for {model!r}")


def default_deployment_call_error(value: object) -> BaseException | None:
    """The error every call to the default deployment raises, from its first reads
    (``_update_kwargs_with_deployment``'s ``deployment["model_name"]``, ``_get_client``'s
    ``deployment["model_info"]["id"]``); None when they read."""
    try:
        served: Final = default_deployment_copy(value, "")
        _ = served["model_name"]  # pyright: ignore[reportIndexIssue]  # read as Python reads it
        _ = served["model_info"]["id"]  # pyright: ignore[reportIndexIssue]  # same
    except Exception as raised:  # noqa: BLE001  # the error the call raises
        return raised
    return None


def served_default_deployment(value: object) -> tuple[Mapping[str, object] | None, bool]:
    """The ``default_deployment`` as router-core serves it, and whether resolving to it raises
    (a value Python cannot copy). A deployment every call to fails is served under
    ``DEFAULT_DEPLOYMENT_FAILING_ID``, one whose id is not a string under
    ``DEFAULT_DEPLOYMENT_COPIED_ID``; names that are not strings are served as text."""
    if value is None:
        return None, False
    try:
        copied: Final = default_deployment_copy(value, "")
    except Exception:  # noqa: BLE001  # any copy error: the pick raises it
        return None, True
    entry: Final = as_str_mapping(copied) or {}
    model_info: Final = dict(as_str_mapping(entry.get("model_info")) or {})
    failing: Final = default_deployment_call_error(value) is not None
    assigned_id: Final = model_info.get("id")
    model_info["id"] = (
        DEFAULT_DEPLOYMENT_FAILING_ID
        if failing
        else assigned_id
        if isinstance(assigned_id, str)
        else DEFAULT_DEPLOYMENT_COPIED_ID
    )
    name: Final = entry.get("model_name")
    return {
        **entry,
        "model_name": name if isinstance(name, str) else str(name or ""),
        "litellm_params": dict(as_str_mapping(entry.get("litellm_params")) or {}),
        "model_info": model_info,
    }, False
