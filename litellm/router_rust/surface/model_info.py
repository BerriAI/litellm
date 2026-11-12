# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportAny=false, reportExplicitAny=false, reportMissingTypeArgument=false, reportAttributeAccessIssue=false, reportTypedDictNotRequiredAccess=false
# (ported from the Python router core as it is written there, over untyped litellm helpers)
"""
``HostModelInfo``: model info, group info, usage reads and response headers for the Rust core's
host registry.

router-core holds the rules: a deployment's model info (``get_router_model_info``,
``get_deployment_model_info``: ``router-core/src/model_info.rs``), a model group's info and the
configured-value reads (``_set_model_group_info``, ``get_model_listing_info``, the configured token
limits, mode, display name and service tiers: ``router-core/src/group_info.rs``), and the usage
counters' keys (``RouterCoreHandle.usage_key``). What lives in this process is answered here as a
query asks for it, at query time: the cost map (``get_model_info``, ``litellm.model_cost``),
provider resolution, the wildcard pattern routes, what the background refresh discovered, and a
group deployment's validated params, supported params, endpoints, fast mode and reasoning efforts.
``ModelGroupInfo`` is built here from the group's first deployment (its validation is the model's)
and takes what router-core's merge assigned.

Still the Python router core's rules here: the model group alias a group-info query resolves
first (the alias map is the registry's live Python objects), the cache of discovered limits, the
configured ``max_input_tokens`` read (a value echoed from the loaded cost map is skipped), and the
response headers.
"""

import builtins
import dataclasses
import math
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, Union, cast  # noqa: TID251  # the core's own signatures, ported unchanged

from pydantic import BaseModel, TypeAdapter
from typing_extensions import overload

import litellm
from litellm._internal_context import service_target
from litellm._logging import verbose_router_logger
from litellm.constants import DEFAULT_MAX_LRU_CACHE_SIZE
from litellm.litellm_core_utils.add_retry_fallback_headers import (
    apply_quality_router_decision_headers,
    apply_remaining_usage_headers,
    apply_response_model_id,
    complexity_router_decision_headers,
    ensure_response_additional_headers,
    prepare_response_for_header_attachment,
)
from litellm.litellm_core_utils.batch_utils import deployment_supported_endpoints
from litellm.litellm_core_utils.core_helpers import as_str_mapping, coerce_token_limit
from litellm.litellm_core_utils.get_model_cost_map import GetModelCostMap
from litellm.litellm_core_utils.litellm_logging import resolve_vertex_location_for_cost
from litellm.litellm_core_utils.reasoning_effort_capability import (
    deployment_is_catalog_mapped,
    intersect_supported_reasoning_efforts,
    resolve_supported_reasoning_efforts,
)
from litellm.litellm_core_utils.routing_common_utils import team_may_use_deployment
from litellm.llms.anthropic.common_utils import AnthropicModelInfo
from litellm.llms.bedrock.common_utils import get_bedrock_base_model
from litellm.redis.stores.router_usage import ROUTER_USAGE_TARGET, RouterUsage, UsageCounter
from litellm.router_python_host import python_host
from litellm.router_utils.model_group_pricing import deployment_pricing
from litellm.router_utils.pre_call_checks.io_token_rate_limit_check import build_io_token_rate_limit_headers
from litellm.router_utils.router_callbacks.track_deployment_metrics import find_deployment_metadata
from litellm.types.router import (
    Deployment,
    DeploymentModelListingInfo,
    DeploymentPricing,
    DeploymentTypedDict,
    DiscoveredDeploymentModelInfo,
    LiteLLM_Params,
    ModelGroupInfo,
)
from litellm.types.utils import COST_MAP_LOOKUP_KEY, ModelInfo, echoed_cost_map_fields
from litellm.types.utils import ModelInfo as ModelMapInfo
from litellm.utils import get_utc_datetime, is_generalized_model_info

if TYPE_CHECKING:
    from litellm.router_rust.surface.registry import GroupCounters
    from litellm.rust_bridge._native import (
        RouterCall,
        RouterCoreHandle,
        _ModelInfoSteps,  # pyright: ignore[reportPrivateUsage]  # the stub's step shapes
    )

_MODEL_INFO_ADAPTER: Final = TypeAdapter(Mapping[str, object])
_LITELLM_PARAMS_ADAPTER: Final = TypeAdapter(Mapping[str, object])

# The key router-core's merge reads as an opaque value (``router-core/src/model_info.rs``).
_OPAQUE_KEY: Final = "__litellm_python_value__"


class _Wire:
    """Values as router-core's model-info merge reads them: plain JSON, with what JSON cannot spell
    (a datetime, a tuple, a non-finite float) passed as an opaque token the merge moves between
    layers without reading, and restored to the same object on the way back."""

    def __init__(self) -> None:
        self._values: Final[list[object]] = []  # mutable-ok: the tokens of one query

    def to_wire(self, value: object) -> object:
        if value is None or isinstance(value, (bool, str)):
            return value
        if isinstance(value, int):
            return value if -(2**63) <= value < 2**64 else self._token(value)
        if isinstance(value, float):
            return value if math.isfinite(value) else self._token(value)
        if isinstance(value, list):
            return [self.to_wire(item) for item in value]
        if isinstance(value, dict) and all(isinstance(key, str) for key in value):
            return {str(key): self.to_wire(item) for key, item in value.items()}
        return self._token(value)

    def from_wire(self, value: object) -> object:
        if isinstance(value, dict):
            if len(value) == 1 and _OPAQUE_KEY in value:
                return self._values[int(value[_OPAQUE_KEY])]
            return {str(key): self.from_wire(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self.from_wire(item) for item in value]
        return value

    def _token(self, value: object) -> Mapping[str, int]:
        self._values.append(value)
        return {_OPAQUE_KEY: len(self._values) - 1}


def _drive_model_info(
    call: "RouterCall[_ModelInfoSteps]", answer: Callable[[Mapping[str, object]], object]
) -> Mapping[str, object]:
    """Drive one model-info merge to its end, answering each lookup with ``answer``. A lookup that
    raises raises out of the query, as it raised in the Python router."""
    outcome: object = None  # rebind-ok: each lookup's answer
    while True:
        step = call.next(outcome)
        if step["kind"] == "done":
            return step
        if step["kind"] == "failed":
            if str(step.get("error_kind")) == "model_info":
                exception: object = getattr(builtins, str(step.get("error_exception")), ValueError)
                raise (exception if isinstance(exception, type) else ValueError)(step.get("error_detail"))
            raise RuntimeError(f"router-core ({step.get('error_kind')}): {step.get('error')}")
        outcome = answer(step)


def _configured_model_info(deployment: Mapping[str, object]) -> Mapping[str, object]:
    return _MODEL_INFO_ADAPTER.validate_python(deployment.get("model_info") or {})


def _configured_deployment_id(deployment: Mapping[str, object]) -> str | None:
    deployment_id: Final = _configured_model_info(deployment).get("id")
    return deployment_id if isinstance(deployment_id, str) else None


def _operator_max_input_tokens(deployment: Mapping[str, object]) -> int | None:
    """``_operator_max_input_tokens``: the deployment's own limit, skipping one a saved
    ``/model/info`` response copied from the cost map."""
    model_info: Final = _configured_model_info(deployment)
    catalog_key: Final = model_info.get(COST_MAP_LOOKUP_KEY)
    catalog_entry: Final = (
        GetModelCostMap.loaded_model_cost_map().get(catalog_key) if isinstance(catalog_key, str) else None
    )
    if catalog_entry is not None and "max_input_tokens" in echoed_cost_map_fields(model_info, catalog_entry):
        return None
    return coerce_token_limit(model_info.get("max_input_tokens"))


# The ModelGroupInfo fields router-core's group merge compares against, as the model validated them.
_GROUP_COMPARED_FIELDS: Final = frozenset(
    {
        "providers",
        "max_input_tokens",
        "max_output_tokens",
        "input_cost_per_token",
        "output_cost_per_token",
        "supports_fast_mode",
        "supported_endpoints",
        "supported_reasoning_efforts",
    }
)
# The ModelGroupInfo fields the Python router assigns tuples to.
_GROUP_TUPLE_FIELDS: Final = frozenset({"supported_endpoints", "supported_reasoning_efforts"})


def _group_field(field: str, value: object) -> object:
    """A merged group field as the Python router assigns it: a tuple field takes the list as a tuple."""
    if field in _GROUP_TUPLE_FIELDS and isinstance(value, list):
        return tuple(cast(list[object], value))  # cast-ok: the merge's list
    return value


@dataclasses.dataclass
class _GroupBuild:
    """One group-info merge as the host answers it: the deployments it reads, each one's validated
    params once its first lookup builds them, and the group built from the first deployment."""

    model_list: "Sequence[dict[str, Any]]"  # mutable-ok: the stored entries, read only
    team_id: str | None
    # Filled as the lookups arrive.
    params: dict[int, LiteLLM_Params] = dataclasses.field(default_factory=dict)  # mutable-ok: see above
    info: ModelGroupInfo | None = None
    pricing: tuple[DeploymentPricing, ...] = ()


def _usage_total(values: Sequence[object]) -> int | None:
    counts: Final = tuple(value for value in values if isinstance(value, int))
    return sum(counts) if counts else None


def _usage_totals(values: Sequence[object], tpm_count: int) -> tuple[int | None, int | None]:
    """The group's (tpm, rpm): each counter Redis or memory holds, summed; none held is ``None``."""
    return _usage_total(values[:tpm_count]), _usage_total(values[tpm_count:])


def _limits_io(info: ModelGroupInfo | None) -> bool:
    return info is not None and (info.itpm is not None or info.otpm is not None)


def _limits_usage(info: ModelGroupInfo | None) -> bool:
    return info is not None and (info.tpm is not None or info.rpm is not None)


def _remaining_usage(
    info: ModelGroupInfo | None,
    io_usage: tuple[int | None, int | None] | None,
    usage: tuple[int | None, int | None] | None,
) -> dict[str, int]:  # mutable-ok: the headers the caller adds to
    """The remaining-limit headers of a group. ITPM/OTPM groups emit input/output token headers, but they may
    also set tpm/rpm, so both sets are built: clients and prometheus gauges that read the standard headers
    still get data."""
    io_headers: Final = (
        {}
        if info is None or io_usage is None
        else build_io_token_rate_limit_headers(
            itpm_limit=info.itpm, otpm_limit=info.otpm, current_itpm=io_usage[0], current_otpm=io_usage[1]
        )
    )
    current_tpm, current_rpm = usage or (None, None)
    tpm_limit: Final = None if info is None else info.tpm
    rpm_limit: Final = None if info is None else info.rpm
    tpm_headers: Final = (
        {}
        if tpm_limit is None
        else {"x-ratelimit-remaining-tokens": tpm_limit - (current_tpm or 0), "x-ratelimit-limit-tokens": tpm_limit}
    )
    rpm_headers: Final = (
        {}
        if rpm_limit is None
        else {
            "x-ratelimit-remaining-requests": rpm_limit - (current_rpm or 0),
            "x-ratelimit-limit-requests": rpm_limit,
        }
    )
    return {**io_headers, **tpm_headers, **rpm_headers}


class HostModelInfo:
    """Mixed into ``HostRegistry``, which holds the model list, the indexes and the host."""

    _core_usage_store: RouterUsage
    _group_counters_cache: dict[str, "GroupCounters"]  # mutable-ok: by model group

    def _live_handle(self) -> "RouterCoreHandle | None": ...  # the registry's: the handle ``RustCore`` set

    def _group_members(self, model_group: str) -> tuple[tuple[object, object], ...] | None: ...  # the registry's

    def _core_handle(self) -> "RouterCoreHandle":
        handle: Final = self._live_handle()
        if handle is None:
            raise RuntimeError("the host registry reads model info through its Rust core")
        return handle

    def _core_get_model_list_of_routed_group(
        self, model_group: str, team_id: str | None = None
    ) -> list:  # mutable-ok: a fresh list, as the core returns it
        if team_id is not None and self._team_deployments_named(team_id, model_group):
            return self._core_get_all_deployments(model_name=model_group, team_id=team_id)
        named: Final = [
            *self._core_get_all_deployments(model_name=model_group),
            *self._core_get_model_list_from_routing_groups(model_name=model_group),
        ]
        return named or self._get_wildcard_deployments(model_name=model_group)

    def _get_wildcard_deployments(
        self, model_name: str, team_id: str | None = None
    ) -> list[dict[str, object]]:  # mutable-ok: a fresh list
        """The deployments a wildcard pattern routes ``model_name`` to, renamed to it (the routing
        runs in router-core, through the registry's pattern router views)."""
        team_router: Final = self._core_team_pattern_routers.get(team_id) if team_id is not None else None
        matches: Final = [
            *self._core_pattern_router.get_deployments_by_pattern(model=model_name),
            *(team_router.get_deployments_by_pattern(model=model_name) if team_router is not None else ()),
        ]
        return [{**DeploymentTypedDict(**m), "model_name": model_name} for m in matches]  # pyright: ignore[reportCallIssue]  # a stored entry

    def _core_get_model_group_info(
        self, model_group: str, *, include_hidden: bool = False, team_id: str | None = None
    ) -> ModelGroupInfo | None:
        """
        For a given model group name, return the combined model info

        Returns:
        - ModelGroupInfo if able to construct a model group
        - None if error constructing model group info or hidden model group (unless include_hidden)
        """
        ## Check if model group alias
        if model_group in self._core_model_group_alias:
            item: Final = self._core_model_group_alias[model_group]
            if isinstance(item, str):
                _router_model_group = item  # rebind-ok: ported from the Python core unchanged
            elif isinstance(item, dict):
                if item["hidden"] is True and not include_hidden:
                    return None
                else:
                    _router_model_group = item["model"]  # rebind-ok: ported from the Python core unchanged
            else:
                return None

            return self._set_model_group_info(
                model_group=_router_model_group,
                user_facing_model_group_name=model_group,
                team_id=team_id,
            )

        ## Check if actual model
        return self._set_model_group_info(
            model_group=model_group, user_facing_model_group_name=model_group, team_id=team_id
        )

    def _set_model_group_info(
        self, model_group: str, user_facing_model_group_name: str, team_id: str | None = None
    ) -> ModelGroupInfo | None:
        """
        For a given model group name, return the combined model info

        Returns:
        - ModelGroupInfo if able to construct a model group
        - None if error constructing model group info

        router-core merges the group (``router-core/src/group_info.rs``); this picks its
        deployments, answers what needs this process (validated params, model info, provider,
        capabilities), builds ``ModelGroupInfo`` from the first deployment and sets what the merge
        assigned on it.
        """
        serves_team_deployments: Final = team_id is not None and bool(
            self._team_deployments_named(team_id, model_group)
        )
        model_list: Final = self._core_get_model_list_of_routed_group(model_group, team_id)
        description: Final[str | None] = next(
            (d for d in (_configured_model_info(m).get("description") for m in model_list) if isinstance(d, str)),
            None,
        )
        # The wildcard check is asked only for a deployment that is not the group's by name.
        pattern_routes: Final = (
            not serves_team_deployments
            and any("model_name" in m and m["model_name"] != model_group for m in model_list)
            and self._core_pattern_router.route(model_group) is not None
        )
        wire: Final = _Wire()
        group: Final = _GroupBuild(model_list=model_list, team_id=team_id)
        call: Final = self._core_handle().model_group_info(
            {
                "model_group": model_group,
                "user_facing_model_group_name": user_facing_model_group_name,
                "serves_team_deployments": serves_team_deployments,
                "pattern_routes": pattern_routes,
                "description": description,
                "deployments": [wire.to_wire(model) for model in model_list],
            }
        )
        done: Final = _drive_model_info(call, partial(self._answer_group_lookup, group=group, wire=wire))
        answer: Final = as_str_mapping(done["value"])
        if answer is None or group.info is None:
            return None
        for field, value in (as_str_mapping(answer["assignments"]) or {}).items():
            setattr(group.info, field, _group_field(field, wire.from_wire(value)))
        group.info.deployment_pricing = group.pricing
        return group.info

    def _answer_group_lookup(self, step: Mapping[str, object], group: "_GroupBuild", wire: "_Wire") -> object:
        """One lookup of a group-info merge: a deployment's facts, or the group's first entry."""
        match step["lookup"]:
            case "group_params":
                index: Final = int(str(step["index"]))
                litellm_params: Final = LiteLLM_Params(**group.model_list[index]["litellm_params"])
                group.params[index] = litellm_params  # rebind-ok: the merge's state, which each lookup advances
                return {
                    "model": litellm_params.model,
                    "configurable_clientside_auth_params": wire.to_wire(
                        litellm_params.configurable_clientside_auth_params
                    ),
                }
            case "group_deployment":
                return self._group_deployment(group, int(str(step["index"])), wire)
            case "group_capabilities":
                deployment: Final = group.model_list[int(str(step["index"]))]
                merged_info: Final[object] = wire.from_wire(step["model_info"])
                model_info: Final = cast(Mapping[str, object], merged_info)  # cast-ok: the merged model info
                litellm_model: Final = str(step["litellm_model"])
                llm_provider: Final = str(step["llm_provider"])
                efforts: Final = resolve_supported_reasoning_efforts(
                    model_info, deployment_is_mapped=step["mapped"] is True
                )
                pricing: Final = deployment_pricing(
                    llm_provider,
                    _configured_deployment_id(deployment),
                    group.params[int(str(step["index"]))].model,
                    model_info,
                    resolve_vertex_location_for_cost(
                        llm_provider, deployment.get("litellm_params"), None, litellm_model
                    ),
                )
                if pricing is not None and pricing not in group.pricing:
                    group.pricing = (*group.pricing, pricing)  # rebind-ok: the merge's state
                return {
                    "supported_endpoints": wire.to_wire(
                        list(
                            deployment_supported_endpoints(
                                model_info.get("supported_endpoints"), deployment, llm_provider
                            )
                        )  # pyright: ignore[reportArgumentType]  # a stored entry, as the core passes it
                        if team_may_use_deployment(_configured_model_info(deployment).get("team_id"), group.team_id)  # pyright: ignore[reportArgumentType]  # the configured team id
                        else []
                    ),
                    "supports_fast_mode": AnthropicModelInfo.supports_fast_mode(litellm_model, llm_provider),
                    "reasoning_efforts": None if efforts is None else list(efforts),
                }
            case "group_seed":
                entry: Final[object] = wire.from_wire(step["info"])
                seed: Final = cast(dict[str, Any], entry)  # cast-ok: the first deployment's entry
                group.info = ModelGroupInfo(**seed)  # rebind-ok: the merge's state, built from the first deployment
                return group.info.model_dump(mode="json", include=set(_GROUP_COMPARED_FIELDS))
            case "reasoning_intersect":
                current: Final = cast("Sequence[str] | None", step["current"])  # cast-ok: plain JSON efforts
                resolved: Final = cast("Sequence[str] | None", step["resolved"])  # cast-ok: plain JSON efforts
                intersected: Final = intersect_supported_reasoning_efforts(current, resolved)
                return None if intersected is None else list(intersected)
            case _:
                return self._answer_model_info_lookup(step, wire)

    def _group_deployment(self, group: "_GroupBuild", index: int, wire: "_Wire") -> Mapping[str, object]:
        """A group deployment's model info (None when it fails), whether the catalog maps it, and its
        provider; with no model info, the provider's supported params."""
        model: Final = group.model_list[index]
        litellm_params: Final = group.params[index]
        model_info_dict: Final = model.get("model_info", {})
        model_info: object
        try:
            model_id: Final = model_info_dict.get("id", None)
            model_info = (  # rebind-ok: one branch binds it
                self._core_get_deployment_model_info(model_id=model_id, model_name=litellm_params.model)
                if model_id is not None
                else None
            )
        except Exception:  # noqa: BLE001  # ported as the core writes it
            model_info = None  # rebind-ok: see above
        deployment_is_mapped: Final = deployment_is_catalog_mapped(model_info, model_info_dict)  # pyright: ignore[reportArgumentType]  # the merged model info, or None
        litellm_model, llm_provider = self._model_group_llm_provider(litellm_params)
        answer: Final[dict[str, object]] = {  # mutable-ok: built here
            "model_info": wire.to_wire(model_info),
            "mapped": deployment_is_mapped,
            "litellm_model": litellm_model,
            "llm_provider": llm_provider,
        }
        if model_info is None:
            answer["supported_openai_params"] = wire.to_wire(
                self._host.invoke("get_supported_openai_params")(model=litellm_model, custom_llm_provider=llm_provider)
            )
        return answer

    def _cached_get_model_group_info(self, model_group: str) -> ModelGroupInfo | None:
        """
        Cached version of get_model_group_info, uses @lru_cache wrapper

        This is a speed optimization, since set_response_headers makes a call to get_model_group_info on every request
        """
        return self._core_get_model_group_info(model_group)

    def _core_cached_model_group_info(self, model_group: str) -> ModelGroupInfo | None:
        return self._cached_get_model_group_info(model_group)

    @overload
    def _core_get_router_model_info(
        self,
        deployment: Union[dict, "Deployment"],  # mutable-ok: ported from the Python core unchanged
        received_model_name: str,
        id: None = None,
    ) -> ModelMapInfo:
        pass

    @overload
    def _core_get_router_model_info(self, deployment: None, received_model_name: str, id: str) -> ModelMapInfo:
        pass

    def _core_get_router_model_info(
        self,
        deployment: Union[dict, "Deployment"] | None,  # mutable-ok: ported from the Python core unchanged
        received_model_name: str,
        id: str | None = None,
    ) -> ModelMapInfo:
        """
        For a given model id, return the model info (max tokens, input cost, output cost, etc.).

        Augment litellm info with additional params set in `model_info`.

        For azure models, ignore the `model:`. Only set max tokens, cost values if base_model is set.
        For every other provider, a base_model naming the same model as `model:` can only widen its context window.

        router-core merges it (``router-core/src/model_info.rs``); this checks the deployment and
        answers the cost-map lookups the merge asks for while it runs.

        Returns
        - ModelInfo - If found -> typed dict with max tokens, input cost, etc.

        Raises:
        - ValueError -> If model is not mapped yet
        """
        if id is not None:
            _deployment: Final = self._core_get_deployment(model_id=id)
            if _deployment is not None:
                deployment = _deployment  # rebind-ok: ported from the Python core unchanged

        if deployment is None:
            raise ValueError("Deployment not found")

        model_info_base_model: Final = (deployment.get("model_info") or {}).get("base_model", None)
        params_base_model: Final = (
            (deployment.get("litellm_params") or {}).get("base_model", None) if model_info_base_model is None else None
        )

        ## GET PROVIDER - reuse LiteLLM_Params if already constructed
        litellm_params_data: Final = deployment.get("litellm_params")
        litellm_params: LiteLLM_Params
        if isinstance(litellm_params_data, LiteLLM_Params):
            litellm_params = litellm_params_data  # rebind-ok: ported from the Python core unchanged
        elif isinstance(litellm_params_data, dict) and "model" in litellm_params_data:
            litellm_params = LiteLLM_Params.model_validate(  # rebind-ok: ported from the Python core unchanged
                litellm_params_data
            )
        else:
            raise ValueError(
                f"Deployment missing valid litellm_params. "
                f"Got: {type(litellm_params_data).__name__}, "
                f"deployment_id: {(deployment.get('model_info') or {}).get('id', 'unknown')}"
            )

        wire: Final = _Wire()
        user_model_info: dict[str, object] = {}  # mutable-ok: filled when the merge asks for it  # rebind-ok: see above

        def read_user_model_info() -> object:
            nonlocal user_model_info
            raw_user_model_info: Final = deployment.get("model_info")
            user_model_info = dict(  # rebind-ok: see above
                _MODEL_INFO_ADAPTER.validate_python(
                    raw_user_model_info.model_dump(exclude_none=True)
                    if isinstance(raw_user_model_info, BaseModel)
                    else raw_user_model_info or {}
                )
            )
            return wire.to_wire(user_model_info)

        def answer(step: Mapping[str, object]) -> object:
            if step["lookup"] == "provider":
                _model, custom_llm_provider, _, _ = self._host.invoke("get_llm_provider")(
                    model=litellm_params.model,
                    litellm_params=litellm_params,
                )
                return [_model, custom_llm_provider]
            if step["lookup"] == "user_model_info":
                return read_user_model_info()
            return self._answer_model_info_lookup(step, wire)

        query: Final = {
            "deployment_id": wire.to_wire((deployment.get("model_info") or {}).get("id")),
            "model_info_base_model": wire.to_wire(model_info_base_model),
            "params_base_model": wire.to_wire(params_base_model),
            "params_model": litellm_params.model,
            "extra_base_model": wire.to_wire((litellm_params.model_extra or {}).get("base_model")),
            "received_model_name": received_model_name,
        }
        done: Final = _drive_model_info(self._core_handle().router_model_info(query), answer)
        merged: Final = wire.from_wire(done["info"])
        if not isinstance(merged, dict):
            return merged  # pyright: ignore[reportReturnType]  # the core returns get_model_info's answer as is
        # The deployment's own values are its objects, as the Python router merges them.
        for key in cast(list[str], done["user_keys"]):  # cast-ok: router-core's key list
            merged[key] = user_model_info[key]
        return merged  # pyright: ignore[reportReturnType]  # the merged map, as the core builds it

    def _core_get_deployment_model_info(self, model_id: str, model_name: str) -> ModelInfo | None:
        """
        For a given model id, return the model info

        1. Check if model_id is in model info
        2. If not, check if litellm model name is in model info
        3. If not, return None

        router-core merges it (``router-core/src/model_info.rs``); this answers its lookups.
        """
        wire: Final = _Wire()
        call: Final = self._core_handle().deployment_model_info(model_id, wire.to_wire(model_name))
        done: Final = _drive_model_info(call, partial(self._answer_model_info_lookup, wire=wire))
        return wire.from_wire(done["info"])  # pyright: ignore[reportReturnType]  # the merged map, as the core builds it

    def _answer_model_info_lookup(self, step: Mapping[str, object], wire: "_Wire") -> object:
        """One cost-map lookup of a model-info merge, answered from this process."""
        match step["lookup"]:
            case "model_info":
                return self._model_info_row(wire.from_wire(step["model"]), wire, step)
            case "cost_entry":
                key: Final = str(step["key"])
                model_cost: Final = litellm.model_cost if step["process"] else self._host.dynamic().model_cost
                return {"entry": wire.to_wire(model_cost.get(key))} if key in model_cost else None
            case "pattern_models":
                return self._pattern_models(str(step["name"]), wire)
            case "bedrock_base":
                return get_bedrock_base_model(str(step["model_id"]))
            case "discovered":
                return wire.to_wire(dict(self._core_get_discovered_model_info(wire.from_wire(step["id"]))))  # pyright: ignore[reportArgumentType]  # the deployment's id, as configured
            case "log":
                log: Final = verbose_router_logger.error if step["level"] == "error" else verbose_router_logger.debug
                log("%s", step["message"])
                return None
            case other:
                raise ValueError(f"router-core asked an unknown model-info lookup {other!r}")

    def _model_info_row(self, model: object, wire: "_Wire", step: Mapping[str, object]) -> object:
        """``get_model_info(model=...)`` as a lookup answers it: the row, ``{"failed": True}`` when a
        tolerated lookup raises, and for an exact lookup whether a generalization rule made it."""
        # The base model's own window is read through the process host, as the Python router reads it.
        host: Final = python_host if step["exact"] else self._host
        try:
            row: Final = host.invoke("get_model_info")(model=model)
        except Exception:  # noqa: BLE001  # get_model_info raises ModelNotMappedError or a bare Exception for an unmapped model
            if step["tolerate"]:
                return {"failed": True}
            raise
        if step["exact"]:
            return {"row": wire.to_wire(row), "generalized": is_generalized_model_info(row)}
        return wire.to_wire(row)

    def _pattern_models(self, name: str, wire: "_Wire") -> object:
        """The ``{id, model}`` of each deployment a wildcard pattern routes ``name`` to."""
        potential_models: Final = self._core_pattern_router.route(name)
        if potential_models is None:
            return None
        routed: Final[list[object]] = []  # mutable-ok: built here
        for potential_model in potential_models:
            try:
                routed.append(
                    {
                        "id": wire.to_wire((potential_model.get("model_info") or {}).get("id")),
                        "model": wire.to_wire((potential_model.get("litellm_params") or {}).get("model")),
                    }
                )
            except Exception:  # noqa: BLE001  # ported as the core writes it
                verbose_router_logger.debug("model info lookup failed; ignored as the core ignores it")
        return routed

    def _core_get_discovered_model_info(self, model_id: str | None) -> Mapping[str, int]:
        cached: Final[object] = self._discovered_model_info_cache.get_cache(model_id)
        if (
            model_id is not None
            and isinstance(cached, DiscoveredDeploymentModelInfo)
            and cached.deployment is self._core_get_model_info(model_id)
        ):
            configured: Final = TypeAdapter(Mapping[str, object]).validate_python(cached.deployment["model_info"])
            return MappingProxyType({key: value for key, value in cached.limits.items() if configured.get(key) is None})
        return MappingProxyType({})

    def _core_get_model_listing_info(self, model_name: str) -> DeploymentModelListingInfo | None:
        """
        Return what the concrete deployments behind model_name contribute to its
        /v1/models entry: the cost-map keys for their underlying models, plus the widest
        configured or discovered token limits. Resolved via O(1) index
        lookup.

        Returns None for wildcard-expanded or unknown names, where the listed name is the
        real model name and no deployment-specific information exists, and treats a
        malformed configured limit as absent rather than failing the listing.

        The whole group is read rather than just its first deployment, so a group that
        mixes models does not advertise a window that depends on config order; the widest
        one is reported, which is what get_model_group_info already shows the Admin UI.
        Keys are deduplicated, so the ordinary group of interchangeable deployments of one
        model still costs the caller a single cost-map lookup. Unlike get_model_group_info,
        this never triggers pattern matching or deep copies, so it is safe to call per
        listed model on the /v1/models hot path.

        router-core combines the deployments (``router-core/src/group_info.rs``).
        """
        deployments: Final = self._core_deployments_named(model_name)
        if not deployments:
            return None
        wire: Final = _Wire()
        call: Final = self._core_handle().model_listing_info([wire.to_wire(deployment) for deployment in deployments])
        done: Final = _drive_model_info(call, partial(self._answer_model_info_lookup, wire=wire))
        listing: Final = as_str_mapping(done["value"]) or {}
        return DeploymentModelListingInfo(
            cost_map_keys=tuple(cast(list[str], listing["cost_map_keys"])),  # cast-ok: router-core's key list
            max_input_tokens=cast("int | None", listing["max_input_tokens"]),  # cast-ok: router-core's limit
            max_output_tokens=cast("int | None", listing["max_output_tokens"]),  # cast-ok: router-core's limit
        )

    def _core_get_configured_token_limits(self, model_name: str) -> "tuple[int | None, int | None]":
        """
        Return (max_input_tokens, max_output_tokens) configured or discovered for a concrete
        deployment of model_name, via O(1) index lookup.

        Returns (None, None) for wildcard-expanded or unknown names, and treats a
        malformed configured value as absent rather than failing the caller.

        Deliberately reads one deployment rather than aggregating the group the way
        get_model_listing_info does: its caller truncates an embedding input to this
        value, so the widest window in a mixed group would be the wrong answer there.
        """
        deployment: Final = self._core_get_deployment_by_model_group_name(model_group_name=model_name)
        if deployment is None:
            return (None, None)
        wire: Final = _Wire()
        configured: Final = {
            key: wire.to_wire(value) for key, value in deployment.model_info.model_dump(exclude_none=True).items()
        }
        call: Final = self._core_handle().configured_token_limits(wire.to_wire(deployment.model_info.id), configured)
        done: Final = _drive_model_info(call, partial(self._answer_model_info_lookup, wire=wire))
        limits: Final = cast("list[int | None]", done["value"])  # cast-ok: router-core's two limits
        return (limits[0], limits[1])

    def _core_get_configured_mode(self, model_name: str) -> "str | None":
        """Return the mode explicitly configured for a concrete deployment."""
        deployment: Final = self._core_get_deployment_by_model_group_name(model_group_name=model_name)
        if deployment is None:
            return None
        return self._core_handle().configured_text(_Wire().to_wire(deployment.model_info.get("mode")))

    def _core_get_configured_display_name(self, model_name: str) -> "str | None":
        """
        Return the display_name explicitly configured in a concrete deployment's
        model_info for model_name, via O(1) index lookup.

        Returns None for wildcard-expanded or unknown names, and treats a
        non-string or empty configured value as absent rather than failing the
        listing. Like get_configured_token_limits, this never triggers pattern
        matching or deep copies, so it is safe to call per listed model on the
        /v1/models hot path.
        """
        deployment: Final = self._core_get_deployment_by_model_group_name(model_group_name=model_name)
        if deployment is None:
            return None
        return self._core_handle().configured_text(_Wire().to_wire(deployment.model_info.get("display_name")))

    def _core_get_configured_service_tiers(self, model_name: str, team_id: str | None = None) -> tuple[object, ...]:
        """
        Return the service_tiers value each deployment a request from team_id
        to model_name can route to (see _routable_deployments) configures in
        its model_info, unvalidated and in model_list order, None for a
        deployment that sets none; the caller validates the shape it expects
        and decides how the deployments combine.

        Returns an empty tuple for wildcard-expanded or unknown names.
        """
        wire: Final = _Wire()
        tiers: Final = self._core_handle().configured_service_tiers(
            [
                wire.to_wire(dict(_configured_model_info(deployment)))
                for deployment in self._routable_deployments(model_name, team_id)
            ]
        )
        return tuple(wire.from_wire(tier) for tier in tiers)

    def _core_get_pinned_reasoning_efforts(self, model_name: str, team_id: str | None = None) -> tuple[object, ...]:
        """
        Return the litellm_params.reasoning_effort each deployment a request
        from team_id to model_name can route to (see _routable_deployments)
        pins, unvalidated and in model_list order, None for a deployment that
        pins none; the caller decides how the deployments combine.

        Returns an empty tuple for wildcard-expanded or unknown names.
        """
        return tuple(
            _LITELLM_PARAMS_ADAPTER.validate_python(deployment["litellm_params"]).get("reasoning_effort")
            for deployment in self._routable_deployments(model_name, team_id)
        )

    def _core_get_configured_max_input_tokens(
        self, model_name: str, team_id: str | None = None
    ) -> tuple[int | None, ...]:
        """
        Return the max_input_tokens each deployment a request from team_id to
        model_name can route to (see _routable_deployments) sets in its own
        model_info, in model_list order, None for a deployment that sets none
        or a malformed one. Unlike get_model_listing_info, neither the cost map
        nor a discovered limit is read, so only an operator's own value counts,
        and a value a saved /model/info response copied from the cost map is
        skipped too.

        Returns an empty tuple for wildcard-expanded or unknown names.
        """
        return tuple(
            _operator_max_input_tokens(deployment) for deployment in self._routable_deployments(model_name, team_id)
        )

    def _group_usage_counters(self, model_group: str) -> tuple[tuple[UsageCounter, ...], ...] | None:
        """This minute's tpm and rpm counters of every deployment in ``model_group`` (can be a wildcard route),
        under the model sent to ``litellm.completion()``, as the usage writes name them."""
        current_minute: Final = get_utc_datetime().strftime("%H-%M")  # use the same timezone regardless of system clock
        members: Final = self._group_members(model_group)
        if members is None:  # no matching deployments
            return None
        cached: Final = self._group_counters_cache.get(model_group)
        if cached is not None and cached[0] is members and cached[1] == current_minute:
            return cached[2]
        deployments: Final = tuple(pair for pair in members if pair[0] is not None and pair[1] is not None)
        counters: Final = tuple(
            tuple(UsageCounter(id, litellm_model, current_minute, kind) for id, litellm_model in deployments)  # pyright: ignore[reportArgumentType]  # what the listing names, as the Python core passed it
            for kind in ("tpm", "rpm")
        )
        if len(self._group_counters_cache) >= DEFAULT_MAX_LRU_CACHE_SIZE:
            self._group_counters_cache.clear()
        self._group_counters_cache[model_group] = (members, current_minute, counters)
        return counters

    async def _core_get_model_group_usage(self, model_group: str) -> tuple[int | None, int | None]:
        """Current (tpm, rpm) usage of ``model_group``."""
        counters: Final = self._group_usage_counters(model_group)
        if counters is None:
            return None, None
        tpm, rpm = counters
        return _usage_totals(await self._core_usage_store.get_many((*tpm, *rpm)), len(tpm))

    def _core_get_model_group_usage_sync(self, model_group: str) -> tuple[int | None, int | None]:
        """``_core_get_model_group_usage`` for a sync call, which has no event loop to await the store on."""
        counters: Final = self._group_usage_counters(model_group)
        if counters is None:
            return None, None
        tpm, rpm = counters
        return _usage_totals(self._core_usage_store.get_many_sync((*tpm, *rpm)), len(tpm))

    async def get_model_group_io_token_usage(self, model_group: str) -> tuple[int | None, int | None]:
        """
        Returns current ITPM/OTPM usage for a model group (sum across deployments).
        """
        dt: Final = get_utc_datetime()
        current_minute: Final = dt.strftime("%H-%M")
        itpm_keys: Final[list[str]] = []  # mutable-ok: ported from the Python core unchanged
        otpm_keys: Final[list[str]] = []  # mutable-ok: ported from the Python core unchanged

        model_list: Final = self._core_get_model_list(model_name=model_group)
        if model_list is None:
            return None, None

        for model in model_list:
            model_id: str | None = model.get("model_info", {}).get("id")
            litellm_model: str | None = model["litellm_params"].get("model")
            if model_id is None or litellm_model is None:
                continue
            itpm_keys.append(self._core_handle().usage_key(model_id, litellm_model, "itpm", current_minute))
            otpm_keys.append(self._core_handle().usage_key(model_id, litellm_model, "otpm", current_minute))

        with service_target(ROUTER_USAGE_TARGET):
            combined_values: Final = await self._core_cache.async_batch_get_cache(keys=itpm_keys + otpm_keys)
        if combined_values is None:
            return None, None

        itpm_values: Final = combined_values[: len(itpm_keys)]
        otpm_values: Final = combined_values[len(itpm_keys) :]

        total_itpm: int | None = None  # rebind-ok: ported from the Python core unchanged
        for value in itpm_values:
            if isinstance(value, int):
                total_itpm = (total_itpm or 0) + value

        total_otpm: int | None = None  # rebind-ok: ported from the Python core unchanged
        for value in otpm_values:
            if isinstance(value, int):
                total_otpm = (total_otpm or 0) + value

        return total_itpm, total_otpm

    async def _core_get_remaining_model_group_usage(
        self, model_group: str
    ) -> dict[str, int]:  # mutable-ok: ported from the Python core unchanged
        info: Final = self._cached_get_model_group_info(model_group)
        io_usage: Final = await self.get_model_group_io_token_usage(model_group) if _limits_io(info) else None
        usage: Final = await self._core_get_model_group_usage(model_group) if _limits_usage(info) else None
        return _remaining_usage(info, io_usage, usage)

    def _core_get_remaining_model_group_usage_sync(
        self, model_group: str, read_io_usage: Callable[[str], tuple[int | None, int | None]]
    ) -> dict[str, int]:  # mutable-ok: what the async read returns
        """``_core_get_remaining_model_group_usage`` for a sync call; ``read_io_usage`` reads the routing cache."""
        info: Final = self._cached_get_model_group_info(model_group)
        io_usage: Final = read_io_usage(model_group) if _limits_io(info) else None
        usage: Final = self._core_get_model_group_usage_sync(model_group) if _limits_usage(info) else None
        return _remaining_usage(info, io_usage, usage)

    async def set_response_headers(
        self,
        response: object,
        model_group: str | None = None,
        request_kwargs: dict[str, object] | None = None,  # mutable-ok: ported from the Python core unchanged
    ) -> Any:  # noqa: ANN401  # the core's signature, ported unchanged
        """
        Add the most accurate rate limit headers for a given model response.

        ## TODO: add model group rate limit headers
        # - if healthy_deployments > 1, return model group rate limit headers
        # - else return the model's rate limit headers
        """
        response = prepare_response_for_header_attachment(response)  # rebind-ok: ported from the Python core unchanged
        if response is None:
            return response

        additional_headers: Final = ensure_response_additional_headers(response)
        apply_response_model_id(
            response, find_deployment_metadata(request_kwargs) if request_kwargs is not None else None
        )
        additional_headers["x-litellm-model-group"] = model_group
        apply_quality_router_decision_headers(additional_headers, request_kwargs)
        additional_headers.update(complexity_router_decision_headers(request_kwargs))

        if model_group is not None:
            remaining_usage: Final = await self._core_get_remaining_model_group_usage(model_group)
            apply_remaining_usage_headers(additional_headers, remaining_usage)
        return response
