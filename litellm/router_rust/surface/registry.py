# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAny=false
# (the ported core code reads untyped litellm helpers: PatternMatchRouter, Deployment.to_json, host.invoke)
"""
``HostRegistry``: the Rust core's live ``model_list`` and the indexes ``registry()`` reads.

router-core builds the deployments (``litellm-rust/crates/router-core/src/build.rs``): it walks
the list, decides which values are ``os.environ/`` secret references, assigns the id of a
deployment that declares none and adds one deployment per ``organization``. What needs live Python
objects is a host step this registry answers, one deployment at a time (``_answer_build_step``):
``secrets`` (``get_secret``), ``validate`` (the PTU, access-window and strict param checks, and the
pricing each deployment writes into ``litellm.model_cost``), ``provider`` (``get_llm_provider``)
and ``added`` (pass-through credentials, pre-routing strategies, the patterns and indexes below).

router-core holds the deployments. ``_core_model_list`` is ``HostedModelList``
(litellm/router_rust/surface/model_list.py): router-core's records read back through the handle, every
write a changeset (a build serves what it added once it ends, a removal is a delete, any other
write a replace), each deployment one Python object while its record is unchanged.

The indexes are router-core's too (``host_index.rs``), kept by the Python router's history rules:
the id, group and team position maps, ``model_names``, the only-growing ``deployment_names``,
the team public names, the provider default ids and the wildcard pattern routers. The registry
hands them out as the live views of litellm/router_rust/surface/indexes.py, whose writes are changesets.

The lookups run in router-core (``lookup.rs``): the deployments by name (team public name first),
the model list with its alias, routing group and wildcard rows, the ids, names and access groups
it lists, the candidate ids for a route, the routable deployments, the credential deployment picks
and a team model's alternatives. Each answers with rows, positions in the list: this registry
reads its stored entries at them (``_rows``), so identity and in-place reads stay its own, and a
position router-core could not read (an index a caller wrote junk into) is read here and raises
as the Python router's read raised. What stays here needs live Python objects: the alias map and
routing groups a lookup is scoped by (``_list_scope``), the provider a wildcard request is tried
under, pydantic ``Deployment`` values, the credentials, the PTU ceilings, search tool support, a
team model's provider pick (``provider_for_generic_call``) and the SDK clients.

Model info and group info are router-core's rules too (``model_info.rs``, ``group_info.rs``),
answered from this process through ``HostModelInfo`` (litellm/router_rust/surface/model_info.py).

The attribute and method names follow the router core's (``_core_model_list``,
``_core_get_model_list`` ...), so ``litellm.router_api.RouterRegistry`` reads this object as its
router and every ``registry()`` query dispatches the way it does on the Python core.
"""

import copy
import dataclasses
import itertools
import json
import os
import re
import threading
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from functools import lru_cache, partial
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, TypeAlias, cast  # noqa: TID251  # entries as a caller wrote them

from pydantic import BaseModel, TypeAdapter

from litellm import exceptions as litellm_exceptions
from litellm import get_secret_str
from litellm._internal_context import service_target
from litellm._logging import verbose_router_logger
from litellm.caching.dual_cache import DualCache
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.constants import DEFAULT_MAX_LRU_CACHE_SIZE
from litellm.litellm_core_utils.access_windows import access_windows_config_error
from litellm.litellm_core_utils.core_helpers import as_str_mapping, without
from litellm.litellm_core_utils.credential_accessor import CredentialAccessor
from litellm.litellm_core_utils.get_llm_provider_logic import (
    declared_authenticating_provider,
    declared_provider,
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
from litellm.redis.stores.connection import connect
from litellm.redis.stores.router_usage import RouterUsage, RouterUsageStore
from litellm.responses.bridge_check import keep_registered_mode
from litellm.router_api import RouterRegistry, deployment_id
from litellm.router_rust.invoke.shaping import ROUTER_CLIENTS_TARGET
from litellm.router_rust.surface.indexes import HostedIndexes
from litellm.router_rust.surface.model_info import HostModelInfo
from litellm.router_rust.surface.model_list import HostedModelList
from litellm.router_rust.surface.replay import live_registries
from litellm.router_strategy.auto_router.auto_router_model_naming import classify_strategy_router_model
from litellm.router_utils.deployment_numbers import refuse_unservable_numbers, refuse_wide_numbers
from litellm.router_utils.endpoint_eligibility import warn_on_unenforced_endpoint_flag
from litellm.router_utils.pre_call_checks.io_token_rate_limit_check import deployment_has_io_token_limits
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
    validate_litellm_params_strictly,
)
from litellm.types.utils import (
    DEPLOYMENT_SCOPED_TOKEN_LIMIT_FIELDS,
    SHARED_ONLY_WHEN_TRUE_CAPABILITY_FIELDS,
    CustomPricingLiteLLMParams,
    LlmProviders,
    LlmProvidersSet,
    is_server_derived_pricing_key,
    shared_backend_model_info,
)
from litellm.utils import (
    ModelInfoLookupMiss,
    catalog_with_runtime_registrations,
    get_potential_model_names,
    get_secret,
    lookup_model_info_for_registration,
    model_cost_registration_key,
    model_info_with_pricing_overrides,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.ptu_shares import PTUTeamCeiling
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
    from litellm.router_python_host import PythonHost
    from litellm.router_rust.steps.callbacks import OptionalChecks
    from litellm.router_rust.steps.prerouting import PreRoutingStrategies
    from litellm.router_strategy.budget_limiter import RouterBudgetLimiting
    from litellm.rust_bridge._native import RouterCoreHandle
    from litellm.types.utils import ModelInfo as ModelMapInfo

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
_TeamIndex: TypeAlias = dict[tuple[str, str], list[int]]

# How a build's deployments join router-core's registry (`build::Install`).
BuildInstall: TypeAlias = Literal["replace", "append", "upsert"]
# Runs one router-core build (`RouterCoreHandle.build`): the entries, the path, whether it resolves
# secrets, how it installs, and the registry's answer to each step. ``RustCore`` supplies it.
BuildRunner: TypeAlias = Callable[
    [Sequence[object], str, bool, BuildInstall, Callable[[Mapping[str, object]], object]], None
]


@dataclasses.dataclass(slots=True)
class _BuildEntry:
    """One ``set_model_list`` entry as its loop holds it (the router core pops ``model_name``,
    ``litellm_params`` and ``model_info`` off the entry and keeps the rest), which router-core's
    build steps name by index."""

    extras: UntypedEntry
    model_name: str
    litellm_params: UntypedEntry
    model_info: UntypedEntry
    declared_id: str | None
    # `organization` is a list, which router-core adds one deployment for each of.
    per_organization: bool = False
    # A retried entry keeps whether its id was declared twice in the list it came from.
    duplicate: bool | None = None


@dataclasses.dataclass(slots=True)
class _Build:
    """The state of one router-core build, which ``_answer_build_step`` reads and writes."""

    path: Literal["create", "add"]
    # `create`: the entries (a deep copy, which the build consumes), and each one once router-core
    # reaches it.
    raw: Sequence[UntypedEntry] = ()
    # Filled as the build reaches each entry.
    entries: dict[int, _BuildEntry] = dataclasses.field(default_factory=dict)  # mutable-ok: filled in step order
    # `add`: the deployment to add, and the entry it is stored as.
    deployment: Deployment | None = None
    stored: UntypedEntry | None = None
    # The deployment the build is on, its provider and its data-source values.
    pending: Deployment | None = None
    provider: str | None = None
    data_source_values: tuple[tuple[int, str, str], ...] = ()
    # `add`: the deployment it added.
    result: Deployment | None = None
    # `create`: the error a deployment raised. The Python core's loop stops there, so every later
    # step is answered as invalid and the build ends, then this is raised.
    failure: Exception | None = None

    def reached_entries(self) -> tuple[UntypedEntry, ...]:
        return tuple(
            {"model_name": entry.model_name, "litellm_params": entry.litellm_params, "model_info": entry.model_info}
            for entry in self.entries.values()
        )


def _creating(model_list: Sequence[object]) -> _Build:
    raw: Final[list[UntypedEntry]] = copy.deepcopy(list(model_list))  # pyright: ignore[reportAssignmentType]  # model_list entries  # mutable-ok: a deep copy the build consumes
    return _Build(path="create", raw=raw)


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


def _declared_upstream(params: Mapping[str, object]) -> tuple[str, str | None] | None:
    model: Final = params.get("model")
    provider: Final = params.get("custom_llm_provider")
    if not isinstance(model, str):
        return None
    return model, declared_provider(model, provider if isinstance(provider, str) else None)


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


def _catalog_entry_has_token_limit(catalog_entry: Mapping[str, object], field: str) -> bool:
    return catalog_entry.get(field) is not None or (
        field == "max_tokens" and catalog_entry.get("max_output_tokens") is not None
    )


def _shared_backend_model_info_for_keys(
    catalog: Mapping[str, Mapping[str, object]],
    backend_keys: tuple[str, ...],
    shared_model_info: Mapping[str, object],
) -> Mapping[str, object]:
    """``shared_model_info`` without a deployment's token limit the catalog already sets on one of
    ``backend_keys`` (``_shared_backend_model_info_for_keys``)."""
    catalog_token_limit_fields: Final = frozenset(
        field
        for field in DEPLOYMENT_SCOPED_TOKEN_LIMIT_FIELDS
        if any(_catalog_entry_has_token_limit(catalog.get(key) or {}, field) for key in backend_keys)
    )
    return {field: value for field, value in shared_model_info.items() if field not in catalog_token_limit_fields}


def _catalog_backend_keys(
    model: str, custom_llm_provider: str | None, backend_keys: tuple[str, ...]
) -> tuple[str, ...]:
    """Every name the catalog may list the backend model under. The provider is the declared one:
    resolving it would reach provider auth (``catalog_backend_keys``)."""
    model_provider_prefix: Final = model.partition("/")[0]
    catalog_provider: Final = (
        custom_llm_provider
        if custom_llm_provider is not None
        else model_provider_prefix
        if model_provider_prefix in LlmProvidersSet
        else None
    )
    if catalog_provider is None:
        return (*backend_keys, model)
    potential_names: Final = get_potential_model_names(model, catalog_provider)
    return (
        *backend_keys,
        model,
        potential_names["split_model"],
        potential_names["stripped_model_name"],
        potential_names["combined_model_name"],
        potential_names["combined_stripped_model_name"],
        potential_names["provider_prefixed_model_name"],
        potential_names["region_free_combined_model_name"],
    )


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
        without(model_info, *CustomPricingLiteLLMParams.model_fields)
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
    backend_keys: Final = _backend_cost_map_keys(model=model, custom_llm_provider=custom_llm_provider)
    shared_model_info: Final = _shared_backend_model_info_for_keys(
        catalog=catalog_with_runtime_registrations(),
        backend_keys=_catalog_backend_keys(model, custom_llm_provider, backend_keys),
        shared_model_info=shared_backend_model_info(registered),
    )
    register_model(
        model_cost={key: keep_registered_mode(key, shared_model_info) for key in backend_keys},
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


def _entry_section(entry: Mapping[str, object], section: str) -> Mapping[str, object]:
    value: Final = entry.get(section)
    return _MODEL_INFO_ADAPTER.validate_python(value) if isinstance(value, Mapping) else {}


def _entry_backend(entry: Mapping[str, object]) -> tuple[str, str | None] | None:
    params: Final = _entry_section(entry, "litellm_params")
    model: Final = params.get("model")
    provider: Final = params.get("custom_llm_provider")
    return (model, provider if isinstance(provider, str) else None) if isinstance(model, str) else None


def _entry_backend_keys(entry: Mapping[str, object]) -> frozenset[str]:
    backend: Final = _entry_backend(entry)
    return frozenset() if backend is None else frozenset(_backend_cost_map_keys(*backend))


def _entry_row_keys(entry: Mapping[str, object]) -> frozenset[str]:
    return frozenset(model_cost_registration_key(key) for key in _entry_backend_keys(entry))


def _shared_capabilities_set_true(entry: Mapping[str, object]) -> frozenset[str]:
    model_info: Final = _entry_section(entry, "model_info")
    return frozenset(field for field in SHARED_ONLY_WHEN_TRUE_CAPABILITY_FIELDS if model_info.get(field) is True)


def _distinct_withdrawals(entries: Iterable[Mapping[str, object]]) -> tuple[Mapping[str, object], ...]:
    return tuple(
        {(_entry_backend_keys(entry), _shared_capabilities_set_true(entry)): entry for entry in entries}.values()
    )


def _row_sets_true(host: "PythonHost", key: str, fields: frozenset[str]) -> bool:
    row: Final[object] = host.dynamic().model_cost.get(model_cost_registration_key(key))
    return isinstance(row, dict) and any(
        _MODEL_INFO_ADAPTER.validate_python(row).get(field) is True for field in fields
    )


def _catalog_sets_true(catalog: Mapping[str, Mapping[str, object]], catalog_keys: tuple[str, ...], field: str) -> bool:
    entry: Final = next((catalog[key] for key in catalog_keys if key in catalog), None)
    return entry is not None and entry.get(field) is True


def withdraw_shared_capabilities(
    host: "PythonHost",
    removed: Mapping[str, object],
    replacement: Mapping[str, object] | None,
    served: Iterable[Mapping[str, object]],
) -> None:
    """For each ``SHARED_ONLY_WHEN_TRUE_CAPABILITY_FIELDS`` flag ``removed`` set ``True``, set the shared
    ``litellm.model_cost`` row(s) ``removed`` registers onto that still hold it ``True`` to whether ``replacement``,
    a ``served`` deployment registering onto one of them, or the catalog still sets it ``True``."""
    backend: Final = _entry_backend(removed)
    set_true: Final = _shared_capabilities_set_true(removed)
    if backend is None or not set_true:
        return
    row_keys: Final = _entry_row_keys(removed)
    kept: Final = (
        _shared_capabilities_set_true(replacement)
        if replacement is not None and _entry_row_keys(replacement) == row_keys
        else frozenset[str]()
    )
    withdrawn: Final = set_true - kept
    held: Final = tuple(key for key in _entry_backend_keys(removed) if _row_sets_true(host, key, withdrawn))
    if not held:
        return
    candidates: Final = tuple(
        entry for entry in served if not withdrawn.isdisjoint(_shared_capabilities_set_true(entry))
    )
    rows_by_backend: Final = {
        keys: frozenset(model_cost_registration_key(key) for key in keys)
        for keys in frozenset(_entry_backend_keys(entry) for entry in candidates)
    }
    siblings_set_true: Final = frozenset(
        itertools.chain.from_iterable(
            _shared_capabilities_set_true(entry) & withdrawn
            for entry in candidates
            if not rows_by_backend[_entry_backend_keys(entry)].isdisjoint(row_keys)
        )
    )
    catalog: Final = catalog_with_runtime_registrations()
    catalog_keys: Final = _catalog_backend_keys(*backend, _backend_cost_map_keys(*backend))
    restored: Final = {
        field: field in siblings_set_true or _catalog_sets_true(catalog, catalog_keys, field) for field in withdrawn
    }
    host.invoke("register_model")(model_cost={key: restored for key in held}, persist_across_reloads=False)


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
    model_info: Final = without(_model_info_of(deployment), "access_groups")
    return {**deployment, "model_info": model_info}


def _ref_env_only(ref: object) -> bool:
    return bool((as_str_mapping(ref) or {}).get("env_only"))


def _duplicate_of(step: Mapping[str, object], entry: _BuildEntry) -> bool:
    """Whether the entry's declared id is declared twice in its list: a retried entry keeps the
    answer its first build had."""
    return entry.duplicate if entry.duplicate is not None else step.get("duplicate_id") is True


def _index_key(value: object) -> str | None:
    """``value`` as a key of the router's name and id indexes: a dict lookup hashes any key and only
    a string can match, so another hashable key finds nothing (``None``), and an unhashable one
    raises as the lookup raises. The native lookups take strings alone."""
    _ = hash(value)
    return value if isinstance(value, str) else None


def _round_trips(value: object) -> bool:
    """Whether JSON reads ``value`` back equal to itself."""
    try:
        loaded: Final[object] = json.loads(json.dumps(value, default=str))  # pyright: ignore[reportAny]  # plain JSON
    except (TypeError, ValueError):
        return False
    return loaded == value


def _has_non_str_keys(value: object) -> bool:
    """Whether ``value`` holds a dict key that is not a string, which a JSON record reads back as one."""
    pending: list[object] = [value]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, Mapping):
            for key, item in current.items():
                if not isinstance(key, str):
                    return True
                pending.append(item)
        elif isinstance(current, list | tuple):
            pending.extend(current)
    return False


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
        # The live model_list: router-core's deployments read back, every write a changeset.
        self._core_model_list: Final[HostedModelList] = HostedModelList(
            self._live_handle, self._hand_changes, self.live_params, self.live_info, lock=self._deployment_index_lock
        )
        # The litellm_params values a deployment's JSON cannot carry (an Exception or model object as
        # ``mock_response``, a callable ``azure_ad_token_provider``), by deployment id: an attempt is called with them, as Python's list held them.
        self._live_params: Final[dict[str, Mapping[str, object]]] = {}  # mutable-ok: written as deployments register
        self._live_info: Final[dict[str, Mapping[str, object]]] = {}  # mutable-ok: written as deployments register
        # The indexes the registry hands out: router-core's, read back, every write a changeset.
        self._indexes: Final = HostedIndexes(self._live_handle, self._hand_changes)
        self._core_deployment_latency_map: dict[str, float] = {}  # mutable-ok: the live latency map
        self._access_groups_cache: Mapping[str, Sequence[str]] | None = None
        # The routing groups by name; RustCore sets them from its resolved settings.
        self._core_routing_groups: dict[str, RoutingGroup] = {}  # mutable-ok: the routing_groups attribute
        self._routing_group_rows: tuple[UntypedEntry, ...] | None = None
        self._provider_unresolved_deployments: tuple[Callable[[], None], ...] = ()
        # Runs router-core's deployment build; RustCore sets it before the first build.
        self.build_runner: BuildRunner | None = None
        self.missing_attribute: Callable[[str], AttributeError] | None = None
        # Hands router-core a write its build did not make (a removal, a swapped list) as a
        # changeset, and reads its current handle; RustCore sets both before the first build.
        self.changes_runner: Callable[[Mapping[str, object]], None] | None = None
        self.handle_lookup: Callable[[], RouterCoreHandle | None] | None = None
        # The pre-routing strategy registries (``registry().plugins``) and the hooks that keep them;
        # RustCore installs its ``PreRoutingStrategies`` (litellm/router_rust/steps/prerouting.py).
        self._core_auto_routers: dict[str, list[TaggedPreRoutingStrategy[object]]] = {}  # mutable-ok: the live registry
        self._core_complexity_routers: dict[str, list[TaggedPreRoutingStrategy[object]]] = {}  # mutable-ok: same
        self._core_adaptive_routers: dict[str, list[TaggedPreRoutingStrategy[object]]] = {}  # mutable-ok: same
        self._core_quality_routers: dict[str, list[TaggedPreRoutingStrategy[object]]] = {}  # mutable-ok: same
        self._core_switchyard_routers: dict[str, list[TaggedPreRoutingStrategy[object]]] = {}  # mutable-ok: same
        self.strategies: PreRoutingStrategies | None = None
        # The router's optional pre-call checks (deployment budgets, prompt caching pins);
        # RustCore installs its ``OptionalChecks`` (litellm/router_rust/steps/callbacks.py).
        self.checks: OptionalChecks | None = None
        # The Rust core's resolution for a request (`_core_get_routable_deployments`), set by the core.
        self.routable_lookup: (
            Callable[[str, Mapping[str, object], str | None], Sequence[Mapping[str, object]]] | None
        ) = None
        # A deployment's cached `max_parallel_requests` limit (litellm/router_rust/invoke/slots.py), set by the core.
        self.max_parallel_requests_lookup: Callable[[Mapping[str, object], Mapping[str, object]], object] | None = None
        # `plugins("routing")`: the router's routing plugins, set by the core.
        self._core_routing_plugins: list[RoutingPlugin] = []  # mutable-ok: the Python core hands out its live list
        # `zero_cost()`: the per-deployment zero-cost lookup the auth checks fill, emptied with the
        # group info caches.
        self._zero_cost_cache: dict[str, bool] = {}  # mutable-ok: the auth checks write into it
        # The routing cache usage reads come from; RustCore sets it to the one its Rust core writes.
        self._core_cache: DualCache = DualCache()
        # The tpm/rpm store usage reads come from; RustCore sets it to the one its Rust core writes.
        self._core_usage_store: RouterUsage = RouterUsageStore(connect(None).router_usage())
        self._discovered_model_info_cache: InMemoryCache = InMemoryCache(
            max_size_in_memory=1, default_ttl=2 * MODEL_INFO_REFRESH_SECONDS
        )
        # The PTU share teams and team ceilings, until the model list next changes.
        self._ptu_cache: _PTUMemo | None = None
        self._cached_get_model_group_info: Callable[[str], ModelGroupInfo | None] = lru_cache(  # pyright: ignore[reportIncompatibleMethodOverride]  # the per-router cache shadows the uncached method, as the core keeps it
            maxsize=DEFAULT_MAX_LRU_CACHE_SIZE
        )(self._core_get_model_group_info)
        self._core_cached_deployment_model_info: Callable[[str, str], ModelMapInfo | None] = lru_cache(
            maxsize=DEFAULT_MAX_LRU_CACHE_SIZE
        )(self._core_get_deployment_model_info)
        self.cached_deployment_tracks_usage: Callable[[str, str], bool] = lru_cache(maxsize=DEFAULT_MAX_LRU_CACHE_SIZE)(
            self._deployment_tracks_usage
        )

    def __getattr__(self, name: str) -> object:
        from litellm.router_rust.core import RustCoreUnsupported

        missing: Final = self.__dict__.get("missing_attribute")
        if name == "_core_model_group_alias" and missing is not None:
            # `del router.model_group_alias`: every read raises, as on the Python core.
            raise missing(name)
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

    def _deployment_tracks_usage(self, deployment_id: str, model_group: str) -> bool:
        """``_deployment_tracks_usage``: a limit on the deployment or in its merged model info, where
        the cost map's own ``tpm`` and ``rpm`` count too."""
        deployment_info: Final = self._core_get_deployment(model_id=deployment_id)
        if deployment_info is None:
            return False
        deployment_model_info: Final = self._core_get_router_model_info(
            deployment=deployment_info,
            received_model_name=model_group,
        )
        configured_limits: Final[tuple[object, ...]] = (
            deployment_info.get("tpm", None),
            deployment_info.get("rpm", None),
            deployment_info.litellm_params.tpm,
            deployment_info.litellm_params.rpm,
            deployment_model_info.get("tpm", None),
            deployment_model_info.get("rpm", None),
        )
        return any(limit is not None for limit in configured_limits) or deployment_has_io_token_limits(
            deployment_info.model_dump()
        )

    def _invalidate_model_group_info_cache(self) -> None:
        self._ptu_cache = None
        self._cached_get_model_group_info.cache_clear()  # pyright: ignore[reportFunctionMemberAccess]  # an lru_cache
        self._core_cached_deployment_model_info.cache_clear()  # pyright: ignore[reportFunctionMemberAccess]  # an lru_cache
        self.cached_deployment_tracks_usage.cache_clear()  # pyright: ignore[reportFunctionMemberAccess]  # an lru_cache
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
        """Replace every deployment (``set_model_list``). All or nothing: when an entry raises,
        the deployments served before the call are put back and the error is raised."""
        previous: Final = [copy.deepcopy(dict(entry)) for entry in self._core_model_list]
        attempt: Final = _creating(model_list)
        try:
            self._replace_model_list(attempt)
        except Exception:
            self._replace_model_list(_creating(previous))
            self._withdraw_shared_capabilities(attempt.reached_entries(), self._served_by_live_routers())
            raise
        self._withdraw_shared_capabilities(previous, self._served_by_live_routers())

    def _withdraw_shared_capabilities(
        self, entries: Iterable[Mapping[str, object]], served: Sequence[Mapping[str, object]]
    ) -> None:
        for entry in _distinct_withdrawals(entries):
            withdraw_shared_capabilities(self._host, entry, None, served)

    def _replace_model_list(self, build: _Build) -> None:
        self._discovered_model_info_cache.flush_cache()
        original_model_list: Final = build.raw
        if self.strategies is not None:
            self.strategies.reset()
        self._provider_unresolved_deployments = ()
        self._invalidate_model_group_info_cache()
        self._invalidate_access_groups_cache()
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
        # router-core walks the list (one deployment per organization, secrets resolved first,
        # ids assigned), asking this registry about each deployment in turn; it starts the list
        # and its id, name and team indexes over, and takes the given names as `model_names`.
        self._run_build(build, original_model_list, resolve_secrets=True, install="replace")
        if self.strategies is not None:
            self.strategies.finalize()
        self._on_change()

    # -- the build's steps (router-core's `build::build`) ------------------------------------------

    def _run_build(
        self, build: _Build, entries: Sequence[object], *, resolve_secrets: bool, install: BuildInstall
    ) -> None:
        """Have router-core build ``entries``, answering its steps from this registry."""
        if self.build_runner is None:
            raise RuntimeError("the host registry builds deployments through its Rust core")
        for entry in entries:
            refuse_wide_numbers(entry)
        self._core_model_list.begin_build(replacing=install == "replace")
        try:
            self.build_runner(entries, build.path, resolve_secrets, install, partial(self._answer_shown, build))
        finally:
            self._core_model_list.end_build()
            self._drop_removed_live_values()
        if build.failure is not None:
            # router-core served what it built before the entry that raised (`set_model_list`
            # cleared the list first), so the router is left as the Python core leaves it.
            raise build.failure

    def _answer_shown(self, build: _Build, step: Mapping[str, object]) -> object:
        """A build step's answer; a deployment the ``added`` step hands router-core shows in the
        live list from then on (router-core serves it when the build ends)."""
        answer: Final = self._answer_build_step(build, step)
        stored: Final = as_str_mapping(answer).get("entry") if step["kind"] == "added" else None  # pyright: ignore[reportOptionalMemberAccess]  # an added step answers a mapping
        if isinstance(stored, Mapping):
            self._core_model_list.built(cast(Mapping[str, object], stored))  # cast-ok: the entry it stores
        return answer

    def _answer_build_step(self, build: _Build, step: Mapping[str, object]) -> object:
        """The answer to one of router-core's build steps. A ``set_model_list`` deployment that
        fails is left out when the router ignores invalid deployments (``_create_deployment``),
        and retried on the next price data reload when its provider did not resolve."""
        kind: Final = step["kind"]
        if build.failure is not None:
            # The build already failed: every later step is skipped, a secrets step unresolved.
            refs_left: Final = step.get("refs")
            return [None] * len(refs_left) if kind == "secrets" and isinstance(refs_left, list) else {"invalid": True}  # pyright: ignore[reportUnknownArgumentType]  # only the count is read
        index: Final = int(str(step["index"]))
        refs: Final = step.get("refs")
        if kind == "secrets" and isinstance(refs, list) and not any(_ref_env_only(ref) for ref in refs):
            return self._build_secrets(build, index, refs)
        if kind == "validate":
            # A new deployment: the entry is popped where the loop pops it, outside the error
            # handling, and nothing of the last deployment carries over.
            build.pending = None  # rebind-ok: the build's state, which each step advances
            build.provider = None  # rebind-ok: the build's state, which each step advances
            build.data_source_values = ()  # rebind-ok: the build's state, which each step advances
            if build.path == "create":
                _ = self._build_entry(build, index)
        try:
            if kind == "secrets":
                return self._build_data_sources(build, refs if isinstance(refs, list) else [])
            if kind == "validate":
                return self._build_validate(build, index, step)
            if kind == "provider":
                build.provider = self._build_provider(build)  # rebind-ok: the build's state, which each step advances
                return {"provider": build.provider}
            return self._build_added(build)
        except Exception as e:
            if build.path != "create":
                raise
            if not self.ignore_invalid_deployments:
                build.failure = e  # rebind-ok: raised once router-core ends the build
                return {"invalid": True}
            if isinstance(e, litellm_exceptions.BadRequestError):
                entry: Final = build.entries[index]
                retried: Final = dataclasses.replace(entry, duplicate=_duplicate_of(step, entry))
                self._provider_unresolved_deployments = (
                    *self._provider_unresolved_deployments,
                    partial(self._retry_create, retried),
                )
            verbose_router_logger.exception(
                "Error creating deployment: %s, ignoring and continuing with other deployments.", e
            )
            return {"invalid": True}

    def _build_entry(self, build: _Build, index: int) -> _BuildEntry:
        """Entry ``index`` as ``set_model_list``'s loop holds it, popped off the first time a step
        names it (so a malformed entry raises where the loop raises)."""
        held: Final = build.entries.get(index)
        if held is not None:
            return held
        model: Final = build.raw[index]
        model_name: Final = model.pop("model_name")
        raw_params: Final = model.pop("litellm_params")
        litellm_params: Final[UntypedEntry] = raw_params if isinstance(raw_params, dict) else {}  # pyright: ignore[reportAssignmentType]  # configured params
        model_info: Final = model.pop("model_info", {})
        info_dict: Final[UntypedEntry] = model_info if isinstance(model_info, dict) else {}  # pyright: ignore[reportAssignmentType]  # a configured model_info
        entry: Final = _BuildEntry(
            extras=model,
            model_name=str(model_name),
            litellm_params=litellm_params,
            model_info=info_dict,
            declared_id=None if info_dict.get("id") is None else str(info_dict["id"]),
            per_organization=isinstance(litellm_params.get("organization"), list),
        )
        build.entries[index] = entry  # rebind-ok: the build's state, which each step advances
        return entry

    def _build_secrets(self, build: _Build, index: int, refs: Sequence[object]) -> Sequence[object]:
        """``set_model_list``'s ``get_secret`` for each top-level ``os.environ/`` value router-core
        found, written into the entry's params."""
        entry: Final = self._build_entry(build, index)
        references: Final = tuple(as_str_mapping(ref) or {} for ref in refs)
        values: Final = tuple(get_secret(str(reference["name"])) for reference in references)
        entry.litellm_params.update(
            {str(reference["key"]): value for reference, value in zip(references, values, strict=True)}
        )
        return values

    def _build_data_sources(self, build: _Build, refs: Sequence[object]) -> Sequence[object]:
        """``_add_deployment``'s environment reads for ``dataSources[].parameters``, applied to the
        deployment where it applied them (``_build_added``)."""
        references: Final = tuple(as_str_mapping(ref) or {} for ref in refs)
        build.data_source_values = tuple(  # rebind-ok: the build's state, which each step advances
            (
                int(str(reference["source"])),
                str(reference["key"]),
                os.environ.get(str(reference["name"]).replace("os.environ/", ""), ""),
            )
            for reference in references
        )
        return tuple(value for _, _, value in build.data_source_values)

    def _build_validate(self, build: _Build, index: int, step: Mapping[str, object]) -> Mapping[str, object]:
        if build.path == "add":
            return self._validate_added_deployment(build)
        entry: Final = self._build_entry(build, index)
        routed: Final = as_str_mapping(step["entry"]) or {}
        routed_info: Final = as_str_mapping(routed.get("model_info")) or {}
        if "id" not in entry.model_info:
            # router-core hashes the JSON form; a value JSON reads back changed hashes as the
            # Python core hashes it (`deployment_id` over the params as given).
            entry.model_info["id"] = (
                routed_info.get("id")
                if _round_trips(entry.litellm_params)
                else deployment_id(entry.model_name, entry.litellm_params)
            )
        if entry.per_organization:
            entry.litellm_params["organization"] = (as_str_mapping(routed.get("litellm_params")) or {}).get(
                "organization"
            )
        duplicate: Final = _duplicate_of(step, entry)
        deployment: Final = self._validated_deployment(entry, taken=duplicate)
        if deployment is None:
            return {"skip": True}
        served: Final = with_admin_api_key_secret_reference(deployment)
        build.pending = served  # rebind-ok: the build's state, which each step advances
        # A dump of this core's own: a `mock_response` ModelResponse warns, and only Python's dumps may.
        return {"entry": build.pending.to_json(exclude_none=True, warnings=False)}

    def _validated_deployment(self, entry: _BuildEntry, *, taken: bool) -> Deployment | None:
        """``_create_deployment`` up to ``_add_deployment``: the config checks, the deployment, its
        pricing, and None when the environment gate leaves it out."""
        model_name: Final = entry.model_name
        litellm_params: Final = entry.litellm_params
        model_info: Final = entry.model_info
        config_sourced: Final = model_info.get("db_model") is not True
        identity_error: Final = (
            ptu_identity_error(
                declared_id=entry.declared_id,
                taken=taken,
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
            Deployment(**entry.extras, model_name=model_name, litellm_params=params, model_info=model_info)  # pyright: ignore[reportArgumentType]  # the entry's extra fields, as the core passes them
        )
        # The values as typed: a cost written as "inf", a share written as 1e20.
        refuse_unservable_numbers(deployment.to_json(exclude_none=True, warnings=False))
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
        return deployment

    def _validate_added_deployment(self, build: _Build) -> Mapping[str, object]:
        """``add_deployment`` up to ``_add_deployment``: an id the registry holds is skipped."""
        deployment: Final = build.deployment
        if deployment is None:
            raise RuntimeError("an add build holds its deployment")
        if deployment.model_info.id and self._core_has_model_id(deployment.model_info.id):
            return {"skip": True}
        declared_params: Final = deployment.litellm_params.model_dump(exclude_none=True)
        warn_on_provider_credential_mismatch(model_name=deployment.model_name, litellm_params=declared_params)
        warn_on_ignored_model_capability_flags(model_name=deployment.model_name, litellm_params=declared_params)
        warn_on_unenforced_endpoint_flag(
            model_name=deployment.model_name, model_info=deployment.model_info.model_dump()
        )
        served: Final = with_admin_api_key_secret_reference(without_model_capability_flags(deployment))
        build.pending = served  # rebind-ok: the build's state, which each step advances
        build.stored = served.to_json(exclude_none=True)  # rebind-ok: the build's state, which each step advances
        refuse_unservable_numbers(build.stored)
        return {"entry": build.stored}

    def _build_provider(self, build: _Build) -> str | None:
        """``_add_deployment``'s provider: None for a prompt-management model, else the provider
        ``get_llm_provider`` resolves, refused when litellm does not serve it."""
        deployment: Final = build.pending
        if deployment is None:
            raise RuntimeError("a provider step follows a validated deployment")
        litellm_model: Final = deployment.litellm_params.model
        if isinstance(deployment.litellm_params.drop_params, str):
            verbose_router_logger.warning(
                "model=%s drop_params=%r is not a flag value, treating it as unset",
                deployment.model_name,
                deployment.litellm_params.drop_params,
            )
        dynamic: Final = self._host.dynamic()
        if "/" in litellm_model and litellm_model.split("/")[0] in dynamic.known_custom_logger_compatible_callbacks:
            return None
        if declared_authenticating_provider(litellm_model, deployment.litellm_params.custom_llm_provider) is not None:
            _, provider = self._model_group_llm_provider(deployment.litellm_params)
            return provider
        _, custom_llm_provider, _, _ = self._host.invoke("get_llm_provider")(
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
        return custom_llm_provider

    def _build_added(self, build: _Build) -> Mapping[str, object]:
        """The rest of ``_add_deployment``: the entry router-core stores, which joins the list and
        the indexes (``host_index.rs``) when the step returns."""
        deployment: Final = build.pending
        if deployment is None:
            raise RuntimeError("an added step follows a validated deployment")
        self._register_deployment(deployment, build.provider, build.data_source_values)
        build.pending = None  # rebind-ok: the build's state, which each step advances
        build.data_source_values = ()  # rebind-ok: the build's state, which each step advances
        self._invalidate_model_group_info_cache()
        self._invalidate_access_groups_cache()
        if build.path == "create":
            return {"entry": deployment.to_json(exclude_none=True)}
        entry: Final = build.stored if build.stored is not None else deployment.to_json(exclude_none=True)
        register_deployment_pricing(self._host, deployment)
        build.result = deployment  # rebind-ok: the build's state, which each step advances
        return {"entry": entry}

    def _retry_create(self, entry: _BuildEntry) -> None:
        """A ``set_model_list`` deployment left out when its provider did not resolve, built again
        and added at the end."""
        build: Final = _Build(path="create")
        build.entries[0] = entry
        self._run_build(
            build,
            [
                {
                    **entry.extras,
                    "model_name": entry.model_name,
                    "litellm_params": entry.litellm_params,
                    "model_info": entry.model_info,
                }
            ],
            resolve_secrets=False,
            install="append",
        )

    @staticmethod
    def _model_group_llm_provider(litellm_params: LiteLLM_Params) -> tuple[str, str]:
        try:
            return resolve_model_and_provider_for_metadata(
                model=litellm_params.model, custom_llm_provider=litellm_params.custom_llm_provider
            )
        except litellm_exceptions.BadRequestError as e:
            verbose_router_logger.error("litellm.router.py::get_model_group_info() - %s", e)
            return "", ""

    def _register_deployment(
        self,
        deployment: Deployment,
        custom_llm_provider: str | None,
        data_source_values: Sequence[tuple[int, str, str]],
    ) -> None:
        """``_add_deployment`` after the provider: its data-source values, pass-through
        credentials and pre-routing strategies (router-core registers its names and patterns)."""
        if deployment.litellm_params.rpm is None and getattr(deployment, "rpm", None) is not None:
            deployment.litellm_params.rpm = getattr(deployment, "rpm")  # noqa: B009  # an extra field of the entry  # rebind-ok: the core sets it on the deployment it adds
        if deployment.litellm_params.tpm is None and getattr(deployment, "tpm", None) is not None:
            deployment.litellm_params.tpm = getattr(deployment, "tpm")  # noqa: B009  # an extra field of the entry  # rebind-ok: see above
        live: Final = {
            key: value
            for key, value in {
                **deployment.litellm_params.__dict__,
                **(deployment.litellm_params.__pydantic_extra__ or {}),
            }.items()
            # Values JSON cannot carry: errors, callables, a str subclass that marks the value (an
            # admin's secret reference, which only the deployment's own key may resolve), and any
            # other value JSON reads back changed (a set, a tuple, a datetime), which the Python
            # core keeps as given. A model is read back as its dump, as on the Python core.
            if isinstance(value, BaseException)
            or callable(value)
            or (isinstance(value, str) and type(value) is not str)
            or not (isinstance(value, BaseModel) or _round_trips(value))
        }
        # model_info values JSON reads back changed: a dict with keys that are not strings.
        live_info: Final = {
            key: value
            for key, value in {
                **deployment.model_info.__dict__,
                **(deployment.model_info.__pydantic_extra__ or {}),
            }.items()
            if _has_non_str_keys(value)
        }
        model_id: Final = deployment.model_info.id
        if model_id is not None:
            if live:
                self._live_params[model_id] = live
            else:
                self._live_params.pop(model_id, None)
            if live_info:
                self._live_info[model_id] = live_info
            else:
                self._live_info.pop(model_id, None)
        data_sources: Final = deployment.litellm_params.get("dataSources", []) or []
        for source, key, value in data_source_values:
            data_sources[source]["parameters"][key] = value
        if custom_llm_provider is not None:
            self._initialize_deployment_for_pass_through(deployment, custom_llm_provider)
        if self.strategies is not None:
            self.strategies.register(deployment)

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

    # -- writes -----------------------------------------------------------------------------------

    def add_deployment(self, deployment: Deployment) -> Deployment | None:
        """Add one deployment unless its id exists (``add_deployment``), built by router-core."""
        build: Final = _Build(path="add", deployment=deployment)
        self._run_build(build, [deployment.to_json(exclude_none=True)], resolve_secrets=False, install="upsert")
        if build.result is None:
            return None
        if self.checks is not None:
            self.checks.sync_deployment_budget(build.result)
        self._on_change()
        return build.result

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
                        removed: Final = self._core_model_list[removal_idx]  # pyright: ignore[reportArgumentType]  # raises IndexError for a position past the list, as the pop does
                        self._remove(model_id)
                        withdraw_shared_capabilities(
                            self._host, removed, deployment.to_json(exclude_none=True), self._served_by_live_routers()
                        )
                if self.strategies is not None:
                    self.strategies.release(on_router)
                self._on_change()
            self.add_deployment(deployment=deployment)
            if self.strategies is not None:
                self.strategies.upserted(deployment, on_router)
        except Exception as e:
            if not self.ignore_invalid_deployments:
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

    def _live_handle(self) -> "RouterCoreHandle | None":
        return None if self.handle_lookup is None else self.handle_lookup()

    def _hand_changes(self, changes: Mapping[str, object]) -> None:
        if self.changes_runner is None:
            raise RuntimeError("the host registry writes deployments through its Rust core")
        with self._deployment_index_lock:
            self.changes_runner(changes)
            self._core_model_list.refresh()
            self._drop_removed_live_values()

    def _drop_removed_live_values(self) -> None:
        """Forget the live values of ids no deployment carries any more, so a router that rotates
        ids does not keep the removed deployments' errors and callables."""
        handle: Final = self._live_handle()
        if handle is None or not (self._live_params or self._live_info):
            return
        for held in (self._live_params, self._live_info):
            for model_id in [id for id in held if handle.deployment(id) is None]:
                del held[model_id]

    def refresh(self) -> None:
        """Read router-core's list again after a write (only what was appended, when that is all
        that changed). The index views are read again when an entry point hands them out, which
        keeps one ``add_deployment`` from reading every index back."""
        self._core_model_list.refresh()

    def _remove(self, model_id: str) -> None:
        """Remove the deployment at the id map's position for ``model_id``, every index shifted
        (``_update_deployment_indices_after_removal``, in router-core)."""
        self._hand_changes({"remove": model_id})
        self._invalidate_model_group_info_cache()
        self._invalidate_access_groups_cache()

    # -- the indexes (router-core's, read back as live views) -------------------------------------

    @property
    def model_id_to_deployment_index_map(self) -> dict[str, int]:  # mutable-ok: the live id index
        self._indexes.refresh()
        return cast(dict[str, int], self._indexes.id_map)  # cast-ok: id to position, as router-core keeps it

    @property
    def _core_model_name_to_deployment_indices(self) -> dict[str, list[int]]:  # mutable-ok: the live name index
        self._indexes.refresh()
        return cast(dict[str, list[int]], self._indexes.model_name)  # cast-ok: name to positions

    @_core_model_name_to_deployment_indices.setter
    def _core_model_name_to_deployment_indices(self, value: object) -> None:
        self._indexes.assign("model_name", value)

    @property
    def _core_team_model_to_deployment_indices(
        self,
    ) -> dict[tuple[str, str], list[int]]:  # mutable-ok: the live team index
        self._indexes.refresh()
        return cast(_TeamIndex, self._indexes.team_model)  # cast-ok: (team, public name) to positions

    @_core_team_model_to_deployment_indices.setter
    def _core_team_model_to_deployment_indices(self, value: object) -> None:
        self._indexes.assign("team_model", value)

    @property
    def _core_model_names(self) -> set[str]:  # mutable-ok: the live group set
        self._indexes.refresh()
        return cast(set[str], self._indexes.model_names)  # cast-ok: group names

    @_core_model_names.setter
    def _core_model_names(self, value: object) -> None:
        self._indexes.assign("model_names", value)

    @property
    def _core_deployment_names(self) -> list[str]:  # mutable-ok: the live name list
        self._indexes.refresh()
        return cast(list[str], self._indexes.deployment_names)  # cast-ok: litellm model names

    @_core_deployment_names.setter
    def _core_deployment_names(self, value: object) -> None:
        self._indexes.assign("deployment_names", value)

    @property
    def _core_team_public_model_names(self) -> frozenset[str]:
        self._indexes.refresh()
        return self._indexes.team_public_model_names

    @_core_team_public_model_names.setter
    def _core_team_public_model_names(self, value: object) -> None:
        self._indexes.assign("team_public_model_names", value)

    @property
    def _core_provider_default_deployment_ids(self) -> list[str]:  # mutable-ok: the live id list
        self._indexes.refresh()
        return cast(list[str], self._indexes.provider_default_ids)  # cast-ok: deployment ids

    @_core_provider_default_deployment_ids.setter
    def _core_provider_default_deployment_ids(self, value: object) -> None:
        self._indexes.assign("provider_default_ids", value)

    @property
    def _core_pattern_router(self) -> PatternMatchRouter:
        self._indexes.refresh()
        return self._indexes.pattern_router

    @_core_pattern_router.setter
    def _core_pattern_router(self, value: object) -> None:
        self._indexes.pattern_router.patterns = getattr(value, "patterns", {})  # pyright: ignore[reportAttributeAccessIssue]  # a router a caller assigned

    @property
    def _core_team_pattern_routers(self) -> dict[str, PatternMatchRouter]:  # mutable-ok: the live per-team routers
        self._indexes.refresh()
        return cast(dict[str, PatternMatchRouter], self._indexes.team_pattern_routers)  # cast-ok: team id to its router

    @_core_team_pattern_routers.setter
    def _core_team_pattern_routers(self, value: object) -> None:
        routers: Final = as_str_mapping(value) or {}
        self._indexes.write(
            {
                "team_patterns": [
                    [
                        str(team),
                        [
                            [regex, deployments]
                            for regex, deployments in (as_str_mapping(getattr(router, "patterns", None)) or {}).items()
                        ],
                    ]
                    for team, router in routers.items()
                ]
            }
        )

    def replace_model_list(self, model_list: "list[UntypedEntry]") -> None:  # mutable-ok: the list as assigned
        """``router.model_list = [...]``: the list is swapped as given, without reindexing, and
        router-core takes it whole."""
        self._core_model_list.replace(model_list)
        self._registry_version += 1
        self._on_change()

    def _served_by_other_routers(self) -> tuple[UntypedEntry, ...]:
        others: Final = (tuple(registry._core_model_list) for registry in live_registries() if registry is not self)
        return tuple(itertools.chain.from_iterable(others))

    def _served_by_live_routers(self) -> tuple[UntypedEntry, ...]:
        return (*self._core_model_list, *self._served_by_other_routers())

    def retire_shared_capabilities(self) -> None:
        """Take back the ``SHARED_ONLY_WHEN_TRUE_CAPABILITY_FIELDS`` flags this router's deployments put on shared
        ``litellm.model_cost`` rows that no other live router's deployment still sets (``retire``)."""
        self._withdraw_shared_capabilities(self._core_model_list, self._served_by_other_routers())

    def _core_delete_deployment(self, id: str) -> UntypedEntry | None:
        """Remove one deployment and return its stored entry (``_delete_deployment``)."""
        try:
            with self._deployment_index_lock:
                deployment_idx: Final = self.model_id_to_deployment_index_map.get(id)
                if deployment_idx is None:
                    return None
                item: Final = self._core_model_list[deployment_idx]  # pyright: ignore[reportArgumentType]  # raises IndexError for a position past the list, as the pop does
                self._remove(id)
                withdraw_shared_capabilities(self._host, item, None, self._served_by_live_routers())
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

    def _core_decryptable_reasoning(
        self, model_group: str, request_input: Sequence[object], targets: Sequence[Deployment]
    ) -> tuple[object, ...]:
        """The input ``targets`` can decrypt (``registry().decryptable_reasoning``)."""
        if self.checks is None:
            return self.__getattr__("_core_decryptable_reasoning")  # pyright: ignore[reportReturnType]  # raises RustCoreUnsupported
        return self.checks.decryptable_reasoning(model_group, request_input, targets)

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
        handle: Final = self._live_handle()
        key: Final = _index_key(candidate_id)
        return handle is not None and key is not None and handle.has_model_id(key)

    def _core_get_model_info(self, id: str) -> UntypedEntry | None:
        with self._deployment_index_lock:
            handle: Final = self._live_handle()
            key: Final = _index_key(id)
            idx: Final[object] = None if handle is None or key is None else handle.id_position(key)
            return None if idx is None else self._core_model_list[idx]  # pyright: ignore[reportArgumentType, reportCallIssue]  # a position, or what a caller wrote, read as Python reads it

    def live_params(self, model_id: object) -> Mapping[str, object]:
        """The litellm_params values of deployment ``model_id`` that its JSON record cannot carry."""
        return self._live_params.get(model_id, {}) if isinstance(model_id, str) else {}

    def live_info(self, model_id: object) -> Mapping[str, object]:
        """The model_info values a deployment's JSON record reads back changed, by id."""
        return self._live_info.get(model_id, {}) if isinstance(model_id, str) else {}

    def _core_get_deployment(self, model_id: str) -> Deployment | None:
        entry: Final = self._core_get_model_info(model_id)
        return None if entry is None else Deployment(**entry)  # pyright: ignore[reportArgumentType]  # a stored entry

    def _core_get_deployment_by_model_group_name(self, model_group_name: str) -> Deployment | None:
        with self._deployment_index_lock:
            handle: Final = self._live_handle()
            key: Final = _index_key(model_group_name)
            indices: Final[object] = None if handle is None or key is None else handle.named_positions(key)
            model: Final = self._core_model_list[indices[0]] if indices else None  # pyright: ignore[reportIndexIssue, reportArgumentType, reportCallIssue]  # positions, or what a caller wrote, read as Python reads them
            return None if model is None else Deployment(**model)  # pyright: ignore[reportArgumentType]  # a stored entry

    def _at(self, indices: object) -> tuple[UntypedEntry, ...]:
        """The entries at ``indices`` (what an index holds), read as Python's list reads them."""
        return tuple(self._core_model_list[idx] for idx in indices)  # pyright: ignore[reportGeneralTypeIssues, reportArgumentType, reportCallIssue]  # positions, or what a caller wrote

    def _core_deployments_named(self, model_name: str) -> tuple[UntypedEntry, ...] | None:
        with self._deployment_index_lock:
            handle: Final = self._live_handle()
            key: Final = _index_key(model_name)
            indices: Final[object] = None if handle is None or key is None else handle.named_positions(key)
            return None if indices is None else self._at(indices)

    def team_public_model_names_of(self, team_id: object) -> frozenset[str]:
        handle: Final = self._live_handle()
        if handle is None or not isinstance(team_id, str):
            return frozenset()
        return frozenset(handle.team_public_names_of(team_id))

    def _team_deployments_named(self, team_id: str, public_model_name: str) -> tuple[UntypedEntry, ...] | None:
        with self._deployment_index_lock:
            handle: Final = self._live_handle()
            team: Final = _index_key(team_id)
            public: Final = _index_key(public_model_name)
            indices: Final[object] = (
                None if handle is None or team is None or public is None else handle.team_positions(team, public)
            )
            return None if indices is None else self._at(indices)

    # -- lookups (router-core `lookup.rs`): rows read at the stored entries --------------------

    def _rows(self, rows: object) -> list[UntypedEntry]:  # mutable-ok: a fresh list
        """The stored entries router-core's lookup ``rows`` name, read as Python's list reads them:
        a row it could not read is indexed here, and raises as the Python router's read raised."""
        entries: Final[list[UntypedEntry]] = []  # mutable-ok: filled per row
        for row in cast(Sequence[Mapping[str, object]], rows):  # cast-ok: router-core's lookup rows
            if "raw" in row:
                entries.append(self._core_model_list[row["raw"]])  # pyright: ignore[reportArgumentType, reportCallIssue]  # what a caller wrote, read as Python reads it
            elif "raw_positions" in row:
                entries.extend(self._at(row["raw_positions"]))
            elif "pattern" in row:
                entries.append({**DeploymentTypedDict(**row["pattern"]), "model_name": row["name"]})  # pyright: ignore[reportCallIssue]  # a pattern router's stored entry
            else:
                entries.append(self._row_entry(row))
        return entries

    def _row_entry(self, row: Mapping[str, object]) -> UntypedEntry:
        held: Final[UntypedEntry] = self._core_model_list[int(str(row["at"]))]
        name: Final = row.get("name")
        named: Final = held if name is None else _renamed(held, str(name))
        group: Final = row.get("group")
        if group is None:
            return named
        return _as_routing_group_row(
            apply_routing_group_priority(self._core_routing_groups[str(group)], str(row["member"]), named)  # pyright: ignore[reportArgumentType]  # a stored entry
        )

    def _raise_at(self, raw: object) -> None:
        """Read the list at the row router-core could not read it at, which raises."""
        if raw is not None:
            _ = self._rows([raw])

    def _list_scope(self, model_name: str | None, *, only: str | None = None) -> Mapping[str, object]:
        """What ``get_model_list`` lists beyond the deployments: the aliases (read here, where an
        alias item that is no mapping or lacks ``hidden`` raises as the Python router's read
        does), the callable routing groups and the wildcard requests for ``model_name``."""
        from litellm.litellm_core_utils.get_llm_provider_logic import inferred_provider

        aliases: Final[list[Mapping[str, str]]] = []  # mutable-ok: filled per alias
        alias_map: Final = self._core_model_group_alias or {}
        if model_name is None or model_name in alias_map:
            items: Final[Sequence[tuple[str, object]]] = (
                [(model_name, alias_map[model_name])] if model_name is not None else list(alias_map.items())
            )
            for model_alias, model_value in items:
                if isinstance(model_value, str):
                    aliases.append({"alias": model_alias, "target": model_value})
                elif isinstance(model_value, dict):
                    item = RouterModelGroupAliasItem(**model_value)  # pyright: ignore[reportCallIssue, reportArgumentType]  # an alias item dict
                    if item["hidden"] is True and model_name is None:
                        continue
                    aliases.append({"alias": model_alias, "target": item["model"]})
        groups: Final = [
            {"name": name, "members": list(group.models)}
            for name in (self._core_routing_groups or {})
            if (group := self.get_routing_group(name)) is not None
        ]
        # The provider is inferred only for a name no deployment or alias serves: inferring it for an
        # authenticating provider's name would start its sign-in (`_get_wildcard_deployments`).
        wildcard: Final = (
            []
            if model_name is None
            else [model_name]
            if model_name in alias_map or model_name in self._core_model_names
            else [model_name, f"{inferred_provider(model_name)}/{model_name}"]
        )
        return {"aliases": aliases, "groups": groups, "wildcard": wildcard, "only": only}

    def _lookup(self) -> "RouterCoreHandle":
        handle: Final = self._live_handle()
        if handle is None:
            raise RuntimeError("the host registry's lookups run in its Rust core")
        return handle

    # -- deployments by name ----------------------------------------------------------------------

    @staticmethod
    def _core_should_include_deployment(
        model_name: str, model: Mapping[str, object], team_id: str | None = None
    ) -> bool:
        info: Final = _model_info_of(model)
        if team_id is not None and info.get("team_id") == team_id and model_name == info.get("team_public_model_name"):
            return True
        # A record with no model_name raises KeyError, as the Python core's lookup does.
        if model_name is not None and model["model_name"] == model_name:  # pyright: ignore[reportUnnecessaryComparison]  # the core checks it
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
            rows, teams = self._lookup().lookup_across_teams(model)
            team_deployments: Final = self._rows(rows)
            if teams > 1:
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
        with self._deployment_index_lock:
            key: Final = _index_key(model_name)
            return [] if key is None else self._rows(self._lookup().lookup_all(key, team_id, model_alias))

    def get_routing_group(self, model_name: str) -> RoutingGroup | None:
        """The routing group callable as ``model_name``: a deployment model_name or an alias of the
        same name shadows it."""
        if not self._core_routing_groups:
            return None
        group: Final = self._core_routing_groups.get(model_name)
        handle: Final = self._live_handle()
        if (
            group is None
            or (handle is not None and handle.named_positions(model_name) is not None)
            or model_name in (self._core_model_group_alias or {})
        ):
            return None
        return group

    def _core_routing_group_has_alternatives(self, model_group: str | None) -> bool:
        """A callable routing group whose member union spans more than one deployment."""
        with self._deployment_index_lock:
            if model_group is None:
                return False
            resolved: Final = self._core_get_model_from_alias(model=model_group) or model_group
            group: Final = self.get_routing_group(resolved)
            if group is None:
                return False
            handle: Final = self._lookup()

            def held(member: str) -> Sequence[object]:
                return cast(Sequence[object], handle.named_positions(member) or ())  # cast-ok: positions, or junk

            return sum(len(held(member)) for member in group.models) > 1

    def _core_get_model_list_from_model_alias(
        self, model_name: str | None = None
    ) -> list[UntypedEntry]:  # mutable-ok: a fresh list
        with self._deployment_index_lock:
            return self._rows(
                self._lookup().lookup_model_list(model_name, None, self._list_scope(model_name, only="aliases"))
            )

    def _core_get_model_list_from_routing_groups(self, model_name: str | None = None) -> Sequence[UntypedEntry]:
        """Callable routing groups as model-list rows: each member deployment under the group's name."""
        with self._deployment_index_lock:
            if model_name is None and self._routing_group_rows is not None:
                return self._routing_group_rows
            rows: Final = tuple(
                self._rows(
                    self._lookup().lookup_model_list(model_name, None, self._list_scope(model_name, only="groups"))
                )
            )
            if model_name is None:
                self._routing_group_rows = rows
            return rows

    def _core_get_model_list(
        self, model_name: str | None = None, team_id: str | None = None
    ) -> list[UntypedEntry] | None:  # mutable-ok: a fresh list, as get_model_list returns
        with self._deployment_index_lock:
            if model_name is not None and _index_key(model_name) is None:
                # No index holds it, so `_get_wildcard_deployments` reads it: what the pattern routers raise.
                team_router: Final = self._core_team_pattern_routers.get(team_id) if team_id is not None else None
                return [
                    *self._core_pattern_router.get_deployments_by_pattern(model=model_name),
                    *(team_router.get_deployments_by_pattern(model=model_name) if team_router is not None else ()),
                ]
            return self._rows(self._lookup().lookup_model_list(model_name, team_id, self._list_scope(model_name)))

    def _core_wildcard_route_deployments(self, model: str, team_id: str | None) -> Sequence[Mapping[str, object]]:
        with self._deployment_index_lock:
            from litellm.litellm_core_utils.get_llm_provider_logic import inferred_provider

            if not isinstance(model, str):  # pyright: ignore[reportUnnecessaryIsInstance]  # a caller's non-string model reads the base routers
                team_router: Final = self._core_team_pattern_routers.get(team_id) if team_id is not None else None
                team_owned: Final = (
                    team_router.get_deployments_by_pattern(model=model) if team_router is not None else ()
                )
                return team_owned or self._core_pattern_router.get_deployments_by_pattern(model=model)
            rows: Final = self._lookup().lookup_wildcard_route(
                model, team_id, [model, f"{inferred_provider(model)}/{model}"]
            )
            routed: Final = cast(Sequence[Mapping[str, object]], rows)  # cast-ok: router-core's wildcard rows
            return [cast(Mapping[str, object], row["pattern"]) for row in routed]  # cast-ok: its routed copy

    def _core_wildcard_routed_models(self, model: str, team_id: str | None) -> frozenset[str]:
        team_router: Final = self._core_team_pattern_routers.get(team_id) if team_id is not None else None
        team_owned: Final = (
            team_router.get_models_by_pattern(model=model) if team_router is not None else frozenset[str]()
        )
        return team_owned or self._core_pattern_router.get_models_by_pattern(model=model)

    def _core_config_deployments(self) -> Iterator[Mapping[str, object]]:
        """A generator, as the Python core's: the rows are read when iteration starts."""
        with self._deployment_index_lock:
            rows: Final = self._rows(self._lookup().lookup_config())
        yield from rows

    # -- names ------------------------------------------------------------------------------------

    def _core_get_model_names(self, team_id: str | None = None) -> list[str]:  # mutable-ok: a fresh list
        with self._deployment_index_lock:
            names, raw = self._lookup().lookup_model_names(team_id, self._list_scope(None))
            self._raise_at(raw)
            return names

    def _core_get_model_ids(
        self, model_name: str | None = None, exclude_team_models: bool = False
    ) -> list[str]:  # mutable-ok: a fresh list
        with self._deployment_index_lock:
            if model_name is not None and _index_key(model_name) is None:
                return []
            ids, raw = self._lookup().lookup_model_ids(model_name, exclude_team_models)
            self._raise_at(raw)
            return ids

    # -- resolving a name -------------------------------------------------------------------------

    def _core_get_model_from_alias(self, model: str) -> str | None:
        return resolve_model_group_alias(self._core_model_group_alias, model)

    def _core_routable_model_group(self, model_name: str) -> str:
        target: Final = self._core_get_model_from_alias(model_name)
        return target if target is not None else model_name

    def _core_is_recognized_model(self, model: str) -> bool:
        handle: Final = self._live_handle()
        key: Final = _index_key(model)
        return (
            (handle is not None and key is not None and handle.has_model_name(key))
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
        handle: Final = self._lookup()
        if handle.has_model_name(model_id):
            return model_id
        if self._core_has_model_id(model_id):
            deployment: Final = self._core_get_deployment(model_id=model_id)
            if deployment is not None and deployment.model_name:
                return deployment.model_name
            return model_id
        listed, serving, raw = handle.lookup_name_serving(model_id, self._list_scope(None))
        self._raise_at(raw)
        if not listed:
            return None
        if serving is not None:
            return serving
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
            if _index_key(model_group_name) is None:
                return None
            position, raw = self._lookup().lookup_group_usable_by_team(model_group_name, team_id)
            self._raise_at(raw)
            return None if position is None else Deployment(**self._core_model_list[position])  # pyright: ignore[reportArgumentType]  # a stored entry

    def _get_team_public_name_deployment(self, model_id: str, team_id: str | None) -> Deployment | None:
        with self._deployment_index_lock:
            if team_id is None or _index_key(model_id) is None:
                return None
            position, raw = self._lookup().lookup_team_public_name(model_id, team_id)
            self._raise_at(raw)
            return None if position is None else Deployment(**self._core_model_list[position])  # pyright: ignore[reportArgumentType]  # a stored entry

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
        with self._deployment_index_lock:
            return tuple(
                self._rows(self._lookup().lookup_routable(self._core_routable_model_group(model_name), team_id))
            )

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
        from litellm.router_rust.core import RustCoreUnsupported

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
        with self._deployment_index_lock:
            from litellm.litellm_core_utils.get_llm_provider_logic import inferred_provider

            resolved: Final = self._core_get_model_from_alias(model=model) or model
            group: Final = self.get_routing_group(resolved) if self._core_routing_groups else None
            ids, raw = self._lookup().lookup_candidate_ids(
                resolved,
                team_id,
                None if group is None else list(group.models),
                [resolved, f"{inferred_provider(resolved)}/{resolved}"],
            )
            self._raise_at(raw)
            return frozenset(ids)

    def _core_team_model_has_alternatives(self, deployment_id: str) -> bool:
        """Whether a team deployment's public name has another unpaused deployment of that team."""
        with self._deployment_index_lock:
            found, raw = self._lookup().lookup_team_model_has_alternatives(deployment_id)
            self._raise_at(raw)
            return found

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
        """The deployments routing would consider for ``request_kwargs``, as the Rust core resolves
        them, the endpoint of ``call_type`` included."""
        lookup: Final = self.routable_lookup
        if lookup is None:
            from litellm.router_rust.core import RustCoreUnsupported

            raise RustCoreUnsupported("registry().get_routable_deployments is not served by the Rust router core")
        return tuple(lookup(model, request_kwargs, call_type))

    def _core_get_routable_upstream_model(self, model_name: str, team_id: str | None = None) -> str | None:
        deployment: Final = next(iter(self._routable_deployments(model_name, team_id)), None)
        if deployment is None:
            return None
        model: Final = _params_of(deployment).get("model")
        return None if model is None else str(model)

    def _core_get_routable_upstream_provider(self, model_name: str, team_id: str | None = None) -> str | None:
        """The provider the deployment ``_core_get_routable_upstream_model`` reads declares: its
        ``litellm_params.custom_llm_provider``, else its model's ``provider/`` prefix. None when it declares neither."""
        deployment: Final = next(iter(self._routable_deployments(model_name, team_id)), None)
        if deployment is None:
            return None
        params: Final = _params_of(deployment)
        model: Final = params.get("model")
        provider: Final = params.get("custom_llm_provider")
        return declared_provider(
            model if isinstance(model, str) else None, provider if isinstance(provider, str) else None
        )

    def _core_get_routable_upstreams(
        self, model_name: str, team_id: str | None = None
    ) -> tuple[tuple[str, str | None], ...]:
        """Per routable deployment that names a model: that model and the provider the deployment declares."""
        upstreams: Final = (
            _declared_upstream(_params_of(deployment)) for deployment in self._routable_deployments(model_name, team_id)
        )
        return tuple(upstream for upstream in upstreams if upstream is not None)

    def _core_get_routable_deployment_model_info(
        self, model_name: str, team_id: str | None = None
    ) -> "tuple[ModelMapInfo | None, ...]":
        """Each deployment a request from ``team_id`` to ``model_name`` can route to, read one at a time the way
        cost tracking reads it: its upstream model's cost map entry merged with its own ``model_info``, None where
        neither is known. Nothing is merged across deployments, so a caller can tell a deployment that lacks a
        capability from one that says nothing."""
        return tuple(
            self._routable_deployment_model_info(deployment)
            for deployment in self._routable_deployments(model_name, team_id)
        )

    def _routable_deployment_model_info(self, deployment: Mapping[str, object]) -> "ModelMapInfo | None":
        deployment_id: Final = _model_info_of(deployment).get("id")
        model: Final = _params_of(deployment).get("model")
        if not isinstance(deployment_id, str) or not isinstance(model, str):
            return None
        return self._core_cached_deployment_model_info(deployment_id, model)

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
        with self._deployment_index_lock:
            use_cache: Final = model_name is None and model_access_group is None and team_id is None
            if use_cache and self._access_groups_cache is not None:
                return self._access_groups_cache
            pairs, raw = self._lookup().lookup_access_groups(
                model_name, model_access_group, team_id, self._list_scope(model_name)
            )
            self._raise_at(raw)
            groups: Final[dict[str, list[str]]] = {g: list(n) for g, n in pairs}  # mutable-ok: as a defaultdict
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
