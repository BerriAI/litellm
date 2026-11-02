# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportAny=false, reportExplicitAny=false, reportMissingTypeArgument=false, reportAttributeAccessIssue=false, reportTypedDictNotRequiredAccess=false
# (ported from the Python router core as it is written there, over untyped litellm helpers)
"""
``HostModelInfo``: model info, group info, usage reads and response headers for the Rust core's
host registry. Bridged host-side, to port to router-core: these read the process cost map
(``litellm.model_cost`` through ``get_model_info``) and the routing cache, which live in Python.
The methods are the Python router core's, ported (never imported from it) so every answer matches.
"""

import copy
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any, Final, Union, cast  # noqa: TID251  # the core's own signatures, ported unchanged

from pydantic import BaseModel, TypeAdapter
from typing_extensions import overload

from litellm._internal_context import service_target
from litellm._logging import verbose_router_logger
from litellm.litellm_core_utils.add_retry_fallback_headers import (
    apply_quality_router_decision_headers,
    apply_remaining_usage_headers,
    apply_response_model_id,
    complexity_router_decision_headers,
    ensure_response_additional_headers,
    prepare_response_for_header_attachment,
)
from litellm.litellm_core_utils.batch_utils import deployment_supported_endpoints
from litellm.litellm_core_utils.core_helpers import coerce_token_limit
from litellm.litellm_core_utils.deployment_pricing import deployment_pricing_models
from litellm.litellm_core_utils.reasoning_effort_capability import (
    deployment_is_catalog_mapped,
    intersect_supported_reasoning_efforts,
    resolve_supported_reasoning_efforts,
)
from litellm.litellm_core_utils.routing_common_utils import team_may_use_deployment
from litellm.llms.anthropic.common_utils import AnthropicModelInfo
from litellm.llms.bedrock.common_utils import get_bedrock_base_model
from litellm.router_python_host import python_host
from litellm.router_utils.pre_call_checks.io_token_rate_limit_check import build_io_token_rate_limit_headers
from litellm.router_utils.router_callbacks.track_deployment_metrics import find_deployment_metadata
from litellm.router_utils.routing_read_batch import ROUTER_USAGE_TARGET
from litellm.types.router import (
    CONFIGURABLE_CLIENTSIDE_AUTH_PARAMS,
    Deployment,
    DeploymentModelListingInfo,
    DiscoveredDeploymentModelInfo,
    LiteLLM_Params,
    ModelGroupInfo,
    RouterCacheEnum,
)
from litellm.types.utils import ModelInfo
from litellm.types.utils import ModelInfo as ModelMapInfo
from litellm.utils import get_utc_datetime, is_generalized_model_info

_MODEL_INFO_ADAPTER: Final = TypeAdapter(Mapping[str, object])


def _cost_value_as_float(value: str | float | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _exact_base_model_max_input_tokens(base_model: str) -> int | None:
    try:
        model_info: Final = python_host.invoke("get_model_info")(model=base_model)
    except Exception:  # noqa: BLE001  # get_model_info raises ModelNotMappedError or a bare Exception for an unmapped model
        return None
    if is_generalized_model_info(model_info):
        return None
    return model_info.get("max_input_tokens")


def _provider_model_id(model_name: str, custom_llm_provider: str) -> str:
    model_id: Final = model_name.rsplit("/", 1)[-1]
    return get_bedrock_base_model(model_id) if custom_llm_provider.startswith("bedrock") else model_id


def _names_route_model(base_model: str, route_key: str, custom_llm_provider: str) -> bool:
    route_model_id: Final = _provider_model_id(route_key, custom_llm_provider)
    base_model_id: Final = _provider_model_id(base_model, custom_llm_provider)
    return route_model_id in (base_model_id, base_model_id.split("@", 1)[0])


def _base_model_widened_window(
    route_model_info: ModelMapInfo, base_model: str | None, custom_llm_provider: str
) -> Mapping[str, int]:
    route_max_input_tokens: Final = route_model_info.get("max_input_tokens")
    if base_model is None or route_max_input_tokens is None:
        return MappingProxyType({})
    if not _names_route_model(base_model, route_model_info["key"], custom_llm_provider):
        return MappingProxyType({})
    base_max_input_tokens: Final = _exact_base_model_max_input_tokens(base_model)
    if base_max_input_tokens is None or base_max_input_tokens <= route_max_input_tokens:
        return MappingProxyType({})
    return MappingProxyType({"max_input_tokens": base_max_input_tokens})


def _configured_model_info(deployment: Mapping[str, object]) -> Mapping[str, object]:
    return _MODEL_INFO_ADAPTER.validate_python(deployment.get("model_info") or {})


def _declared_base_model(user_model_info: Mapping[str, object]) -> str | None:
    match user_model_info:
        case {"base_model": str() as base_model}:
            return base_model
        case _:
            return None


class HostModelInfo:
    """Mixed into ``HostRegistry``, which holds the model list, the indexes and the host."""

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

    def _set_model_group_info(  # noqa: C901  # ported as the core writes it
        self, model_group: str, user_facing_model_group_name: str, team_id: str | None = None
    ) -> ModelGroupInfo | None:
        """
        For a given model group name, return the combined model info

        Returns:
        - ModelGroupInfo if able to construct a model group
        - None if error constructing model group info
        """
        model_group_info: ModelGroupInfo | None = None  # rebind-ok: ported from the Python core unchanged

        total_tpm: int | None = None  # rebind-ok: ported from the Python core unchanged
        total_rpm: int | None = None  # rebind-ok: ported from the Python core unchanged
        total_itpm: int | None = None  # rebind-ok: ported from the Python core unchanged
        total_otpm: int | None = None  # rebind-ok: ported from the Python core unchanged
        configurable_clientside_auth_params: CONFIGURABLE_CLIENTSIDE_AUTH_PARAMS = (  # rebind-ok: ported from the Python core unchanged
            None
        )
        reasoning_efforts_initialized = False  # rebind-ok: ported from the Python core unchanged
        reasoning_efforts_unknown = False  # rebind-ok: ported from the Python core unchanged
        serves_team_deployments: Final = team_id is not None and bool(
            self._team_deployments_named(team_id, model_group)
        )
        model_list: Final = self._core_get_model_list_of_routed_group(model_group, team_id)
        model_group_description: Final[str | None] = next(
            (d for d in (_configured_model_info(m).get("description") for m in model_list) if isinstance(d, str)),
            None,
        )
        for model in model_list:
            is_match = False
            if (
                serves_team_deployments
                or "model_name" in model
                and model["model_name"] == model_group
                or "model_name" in model
                and self._core_pattern_router.route(model_group) is not None
            ):  # exact match
                is_match = True

            if not is_match:
                continue
            # model in model group found #
            litellm_params = LiteLLM_Params(**model["litellm_params"])
            # get configurable clientside auth params
            configurable_clientside_auth_params = litellm_params.configurable_clientside_auth_params

            # Cache nested dict access to avoid repeated temporary dict allocations
            model_litellm_params = model.get("litellm_params", {})
            model_info_dict = model.get("model_info", {})

            # get model tpm
            _deployment_tpm: int | None = None
            if _deployment_tpm is None:
                _deployment_tpm = model.get("tpm", None)
            if _deployment_tpm is None:
                _deployment_tpm = model_litellm_params.get("tpm", None)
            if _deployment_tpm is None:
                _deployment_tpm = model_info_dict.get("tpm", None)

            # get model rpm
            _deployment_rpm: int | None = None
            if _deployment_rpm is None:
                _deployment_rpm = model.get("rpm", None)
            if _deployment_rpm is None:
                _deployment_rpm = model_litellm_params.get("rpm", None)
            if _deployment_rpm is None:
                _deployment_rpm = model_info_dict.get("rpm", None)

            _deployment_itpm: int | None = model.get("itpm")
            if _deployment_itpm is None:
                _deployment_itpm = model_litellm_params.get("itpm", None)
            if _deployment_itpm is None:
                _deployment_itpm = model_info_dict.get("itpm", None)

            _deployment_otpm: int | None = model.get("otpm")
            if _deployment_otpm is None:
                _deployment_otpm = model_litellm_params.get("otpm", None)
            if _deployment_otpm is None:
                _deployment_otpm = model_info_dict.get("otpm", None)

            # get model info
            try:
                model_id = model_info_dict.get("id", None)
                if model_id is not None:
                    model_info = self._core_get_deployment_model_info(
                        model_id=model_id, model_name=litellm_params.model
                    )
                else:
                    model_info = None
            except Exception:  # noqa: BLE001  # ported as the core writes it
                model_info = None

            deployment_is_mapped = deployment_is_catalog_mapped(model_info, model_info_dict)

            # get llm provider
            litellm_model, llm_provider = self._model_group_llm_provider(litellm_params)

            if model_info is None:
                supported_openai_params = self._host.invoke("get_supported_openai_params")(
                    model=litellm_model, custom_llm_provider=llm_provider
                )
                if supported_openai_params is None:
                    supported_openai_params = []

                # Get mode from database model_info if available, otherwise default to "chat"
                db_model_info = model.get("model_info", {})
                mode = db_model_info.get("mode", "chat")
                input_cost_per_token = _cost_value_as_float(db_model_info.get("input_cost_per_token"))
                output_cost_per_token = _cost_value_as_float(db_model_info.get("output_cost_per_token"))

                model_info = ModelMapInfo(
                    key=model_group,
                    max_tokens=None,
                    max_input_tokens=None,
                    max_output_tokens=None,
                    input_cost_per_token=input_cost_per_token,
                    output_cost_per_token=output_cost_per_token,
                    litellm_provider=llm_provider,
                    mode=mode,
                    supported_openai_params=supported_openai_params,
                    supports_system_messages=None,
                )

            if model_group_info is None:
                model_group_info = ModelGroupInfo(
                    **{
                        "model_group": user_facing_model_group_name,
                        "providers": [llm_provider],
                        "description": model_group_description,
                        **model_info,
                        "supports_fast_mode": True,
                        "supported_reasoning_efforts": None,
                        "supported_endpoints": None,
                    }
                )
            else:
                # if max_input_tokens > curr
                # if max_output_tokens > curr
                # if input_cost_per_token > curr
                # if output_cost_per_token > curr
                # supports_parallel_function_calling == True
                # supports_vision == True
                # supports_function_calling == True
                if llm_provider not in model_group_info.providers:
                    model_group_info.providers.append(llm_provider)
                if (
                    model_info.get("max_input_tokens", None) is not None
                    and model_info["max_input_tokens"] is not None
                    and (
                        model_group_info.max_input_tokens is None
                        or model_info["max_input_tokens"] > model_group_info.max_input_tokens
                    )
                ):
                    model_group_info.max_input_tokens = model_info["max_input_tokens"]
                if (
                    model_info.get("max_output_tokens", None) is not None
                    and model_info["max_output_tokens"] is not None
                    and (
                        model_group_info.max_output_tokens is None
                        or model_info["max_output_tokens"] > model_group_info.max_output_tokens
                    )
                ):
                    model_group_info.max_output_tokens = model_info["max_output_tokens"]
                _input_cost_per_token = _cost_value_as_float(model_info.get("input_cost_per_token"))
                if _input_cost_per_token is not None and (
                    model_group_info.input_cost_per_token is None
                    or _input_cost_per_token > (model_group_info.input_cost_per_token or 0.0)
                ):
                    model_group_info.input_cost_per_token = _input_cost_per_token
                _output_cost_per_token = _cost_value_as_float(model_info.get("output_cost_per_token"))
                if _output_cost_per_token is not None and (
                    model_group_info.output_cost_per_token is None
                    or _output_cost_per_token > (model_group_info.output_cost_per_token or 0.0)
                ):
                    model_group_info.output_cost_per_token = _output_cost_per_token
                if (
                    model_info.get("supports_parallel_function_calling", None) is not None
                    and model_info["supports_parallel_function_calling"] is True
                ):
                    model_group_info.supports_parallel_function_calling = True
                if model_info.get("supports_vision", None) is not None and model_info["supports_vision"] is True:
                    model_group_info.supports_vision = True
                if (
                    model_info.get("supports_function_calling", None) is not None
                    and model_info["supports_function_calling"] is True
                ):
                    model_group_info.supports_function_calling = True
                if (
                    model_info.get("supports_web_search", None) is not None
                    and model_info["supports_web_search"] is True
                ):
                    model_group_info.supports_web_search = True
                if (
                    model_info.get("supports_url_context", None) is not None
                    and model_info["supports_url_context"] is True
                ):
                    model_group_info.supports_url_context = True

                if model_info.get("supports_reasoning", None) is not None and model_info["supports_reasoning"] is True:
                    model_group_info.supports_reasoning = True
                if (
                    model_info.get("supported_openai_params", None) is not None
                    and model_info["supported_openai_params"] is not None
                ):
                    model_group_info.supported_openai_params = model_info["supported_openai_params"]
                if model_info.get("tpm", None) is not None and _deployment_tpm is None:
                    _deployment_tpm = model_info.get("tpm")
                if model_info.get("rpm", None) is not None and _deployment_rpm is None:
                    _deployment_rpm = model_info.get("rpm")

            model_group_info.supported_endpoints = (
                tuple(
                    dict.fromkeys(
                        (
                            *(model_group_info.supported_endpoints or ()),
                            *(
                                deployment_supported_endpoints(
                                    model_info.get("supported_endpoints"), model, llm_provider
                                )
                                if team_may_use_deployment(_configured_model_info(model).get("team_id"), team_id)
                                else ()
                            ),
                        )
                    )
                )
                or None
            )
            model_group_info.supports_fast_mode = model_group_info.supports_fast_mode and (
                AnthropicModelInfo.supports_fast_mode(litellm_model, llm_provider)
            )
            deployment_reasoning_efforts = resolve_supported_reasoning_efforts(
                model_info, deployment_is_mapped=deployment_is_mapped
            )
            if deployment_reasoning_efforts is None:
                reasoning_efforts_unknown = True
                model_group_info.supported_reasoning_efforts = None
            elif not reasoning_efforts_initialized:
                reasoning_efforts_initialized = True
                if not reasoning_efforts_unknown:
                    model_group_info.supported_reasoning_efforts = deployment_reasoning_efforts
            elif not reasoning_efforts_unknown:
                model_group_info.supported_reasoning_efforts = intersect_supported_reasoning_efforts(
                    model_group_info.supported_reasoning_efforts,
                    deployment_reasoning_efforts,
                )

            if _deployment_tpm is not None:
                if total_tpm is None:
                    total_tpm = 0
                total_tpm += _deployment_tpm

            if _deployment_rpm is not None:
                if total_rpm is None:
                    total_rpm = 0
                total_rpm += _deployment_rpm

            if _deployment_itpm is not None:
                if total_itpm is None:
                    total_itpm = 0
                total_itpm += _deployment_itpm

            if _deployment_otpm is not None:
                if total_otpm is None:
                    total_otpm = 0
                total_otpm += _deployment_otpm
        if model_group_info is not None:
            ## UPDATE WITH TOTAL TPM/RPM FOR MODEL GROUP
            if total_tpm is not None:
                model_group_info.tpm = total_tpm

            if total_rpm is not None:
                model_group_info.rpm = total_rpm

            if total_itpm is not None:
                model_group_info.itpm = total_itpm

            if total_otpm is not None:
                model_group_info.otpm = total_otpm

            ## UPDATE WITH CONFIGURABLE CLIENTSIDE AUTH PARAMS FOR MODEL GROUP
            if configurable_clientside_auth_params is not None:
                model_group_info.configurable_clientside_auth_params = configurable_clientside_auth_params

        return model_group_info

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

    def _core_get_router_model_info(  # noqa: C901  # ported as the core writes it
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

        ## GET BASE MODEL
        base_model = (deployment.get("model_info") or {}).get(  # rebind-ok: ported from the Python core unchanged
            "base_model", None
        )
        if base_model is None:
            base_model = (  # rebind-ok: ported from the Python core unchanged
                deployment.get("litellm_params") or {}
            ).get("base_model", None)

        model = base_model  # rebind-ok: ported from the Python core unchanged

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
        _model, custom_llm_provider, _, _ = self._host.invoke("get_llm_provider")(
            model=litellm_params.model,
            litellm_params=litellm_params,
        )

        ## SET MODEL TO 'model=' - if base_model is None + not azure
        if custom_llm_provider == "azure" and base_model is None:
            # Router init auto-registers every deployment name into
            # litellm.model_cost as a zeroed stub, so membership alone can't
            # tell a resolvable name apart; require usable limits/costs.
            _azure_fallback_key = (  # rebind-ok: ported from the Python core unchanged
                _model if _model.startswith("azure/") else f"azure/{_model}"
            )
            _fallback_entry = self._host.dynamic().model_cost.get(  # rebind-ok: ported from the Python core unchanged
                _azure_fallback_key
            )
            _fallback_resolves = isinstance(  # rebind-ok: ported from the Python core unchanged
                _fallback_entry, Mapping
            ) and (
                (_fallback_entry.get("max_input_tokens") or 0) > 0
                or (_fallback_entry.get("input_cost_per_token") or 0) > 0
            )
            if _fallback_resolves:
                verbose_router_logger.debug(
                    "Azure deployment '%s' has no base_model set; using '%s' from the model cost map for max tokens, cost tracking, etc.",
                    _model,
                    _azure_fallback_key,
                )
            else:
                verbose_router_logger.error(
                    "Could not identify azure model '%s'. Set azure 'base_model' for accurate max tokens, cost tracking, etc.- https://docs.litellm.ai/docs/proxy/cost_tracking#spend-tracking-for-azure-openai-models",
                    _model,
                )
        elif custom_llm_provider != "azure":
            model = _model  # rebind-ok: ported from the Python core unchanged

            if "*" in model:  # only call pattern_router for wildcard models
                potential_models: Final = self._core_pattern_router.route(received_model_name)
                if potential_models is not None:
                    for potential_model in potential_models:
                        try:
                            if (potential_model.get("model_info") or {}).get("id") == (
                                deployment.get("model_info") or {}
                            ).get("id"):
                                model = (potential_model.get("litellm_params") or {}).get("model")
                                break
                        except Exception:  # noqa: BLE001  # ported as the core writes it
                            verbose_router_logger.debug("model info lookup failed; ignored as the core ignores it")

        ## GET LITELLM MODEL INFO - raises exception, if model is not mapped
        if model is None:
            # Handle case where base_model is None (e.g., Azure models without base_model set)
            # Use the original model from litellm_params
            model = _model  # rebind-ok: ported from the Python core unchanged

        if not model.startswith(f"{custom_llm_provider}/"):
            model_info_name = f"{custom_llm_provider}/{model}"  # rebind-ok: ported from the Python core unchanged
        else:
            model_info_name = model  # rebind-ok: ported from the Python core unchanged

        model_info: Final = self._host.invoke("get_model_info")(model=model_info_name)
        if model_info is None:
            return model_info  # pyright: ignore[reportReturnType]  # the core returns get_model_info's answer as is

        ## CHECK USER SET MODEL INFO
        raw_user_model_info: Final = deployment.get("model_info")
        user_model_info: Final = _MODEL_INFO_ADAPTER.validate_python(
            raw_user_model_info.model_dump(exclude_none=True)
            if isinstance(raw_user_model_info, BaseModel)
            else raw_user_model_info or {}
        )

        # get_model_info() hands back an lru_cache'd dict, so merge into a copy; unset
        # values are skipped or Deployment's None pricing defaults would erase the map's
        merged_model_info: Final[ModelMapInfo] = {  # pyright: ignore[reportAssignmentType]  # the merged map, as the core builds it
            **copy.deepcopy(model_info),
            **_base_model_widened_window(
                model_info,
                _declared_base_model(user_model_info) or _declared_base_model(litellm_params.model_extra or {}),
                custom_llm_provider,
            ),
            **self._core_get_discovered_model_info((deployment.get("model_info") or {}).get("id")),
            **MappingProxyType({key: value for key, value in user_model_info.items() if value is not None}),
        }

        return merged_model_info

    def _core_get_deployment_model_info(self, model_id: str, model_name: str) -> ModelInfo | None:
        """
        For a given model id, return the model info

        1. Check if model_id is in model info
        2. If not, check if litellm model name is in model info
        3. If not, return None
        """
        from litellm.utils import cost_map_omits_token_price, update_dictionary

        model_info: ModelInfo | None = None  # rebind-ok: ported from the Python core unchanged
        custom_model_info: (  # mutable-ok: ported from the Python core unchanged  # rebind-ok: ported from the Python core unchanged
            dict | None
        ) = None
        litellm_model_name_model_info: ModelInfo | None = None  # rebind-ok: ported from the Python core unchanged
        base_model_key: str | None = None  # rebind-ok: ported from the Python core unchanged

        try:
            if model_id in self._host.dynamic().model_cost:
                cost_entry: Final = self._host.dynamic().model_cost.get(model_id) or MappingProxyType({})
                if not isinstance(cost_entry, Mapping):
                    raise TypeError(f"model_cost[{model_id!r}] is not a mapping")
                custom_model_info = {  # rebind-ok: ported from the Python core unchanged
                    **copy.deepcopy(cost_entry),
                    **self._core_get_discovered_model_info(model_id),
                }
        except Exception:  # noqa: BLE001  # ported as the core writes it
            verbose_router_logger.debug("model info lookup failed; ignored as the core ignores it")

        try:
            litellm_model_name_model_info = self._host.invoke(  # rebind-ok: ported from the Python core unchanged
                "get_model_info"
            )(model=model_name)
        except Exception:  # noqa: BLE001  # ported as the core writes it
            verbose_router_logger.debug("model info lookup failed; ignored as the core ignores it")

        ## check for base model
        try:
            if custom_model_info is not None:
                base_model: Final = custom_model_info.get("base_model", None)
                if base_model is not None:
                    ## update litellm model info with base model info
                    base_model_info: Final = copy.deepcopy(self._host.invoke("get_model_info")(model=base_model))
                    if base_model_info is not None:
                        base_model_key = base_model_info.get("key")  # rebind-ok: ported from the Python core unchanged
                        # Base model provides defaults, custom model info overrides
                        custom_model_info = update_dictionary(  # rebind-ok: ported from the Python core unchanged
                            cast(dict, base_model_info),  # cast-ok: ported from the Python core unchanged
                            custom_model_info,
                        )
        except Exception:  # noqa: BLE001  # ported as the core writes it
            verbose_router_logger.debug("model info lookup failed; ignored as the core ignores it")

        # Three mutually exclusive scenarios for the model's metadata:
        if custom_model_info is not None and litellm_model_name_model_info is not None:
            # (1) It has both custom model_info set and exists in the built-in map
            # merge with custom overriding built-in
            model_info = cast(  # cast-ok: ported from the Python core unchanged  # rebind-ok: ported from the Python core unchanged
                ModelInfo,
                update_dictionary(
                    copy.deepcopy(
                        cast(dict, litellm_model_name_model_info)  # cast-ok: ported from the Python core unchanged
                    ),
                    custom_model_info,
                ),
            )
        elif litellm_model_name_model_info is not None:
            # (2) Built-in only — no custom pricing to merge
            model_info = copy.deepcopy(  # rebind-ok: ported from the Python core unchanged
                litellm_model_name_model_info
            )
        elif custom_model_info is not None:
            # (3) Custom only — model not in built-in cost map yet
            # custom_model_info already includes base_model defaults at this point, if applicable
            model_info = cast(  # cast-ok: ported from the Python core unchanged  # rebind-ok: ported from the Python core unchanged
                ModelInfo, custom_model_info
            )

        if model_info is None:
            return None
        builtin_key: Final = (
            litellm_model_name_model_info.get("key") if litellm_model_name_model_info is not None else None
        )
        if cost_map_omits_token_price(model_id, builtin_key, base_model_key):
            return cast(  # cast-ok: TypedDict spread with overridden keys loses its type
                ModelInfo, {**model_info, "input_cost_per_token": None, "output_cost_per_token": None}
            )
        return model_info

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
        """
        deployments: Final = self._core_deployments_named(model_name)
        if not deployments:
            return None

        model_infos: Final = tuple(
            MappingProxyType(
                {
                    **self._core_get_discovered_model_info(
                        (deployment.get("model_info") or MappingProxyType({})).get("id")
                    ),
                    **MappingProxyType(
                        {
                            k: v
                            for k, v in (deployment.get("model_info") or MappingProxyType({})).items()
                            if v is not None
                        }
                    ),
                }
            )
            for deployment in deployments
        )
        raw_deployments: Final[tuple[object, ...]] = deployments
        pricing_keys: Final = (deployment_pricing_models(deployment) for deployment in raw_deployments)
        cost_map_keys: Final = tuple(dict.fromkeys(keys[0] for keys in pricing_keys if keys))
        return DeploymentModelListingInfo(
            cost_map_keys=cost_map_keys,
            max_input_tokens=self._widest_configured_limit(model_infos, "max_input_tokens"),
            max_output_tokens=self._widest_configured_limit(model_infos, "max_output_tokens"),
        )

    @staticmethod
    def _widest_configured_limit(model_infos: Sequence[Mapping[str, object]], field: str) -> int | None:
        """The largest usable value of ``field`` across a group's configured model_info blocks."""
        limits: Final = tuple(
            limit
            for limit in (coerce_token_limit(model_info.get(field)) for model_info in model_infos)
            if limit is not None
        )
        return max(limits) if limits else None

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

        model_info: Final = MappingProxyType(
            {
                **self._core_get_discovered_model_info(deployment.model_info.id),
                **deployment.model_info.model_dump(exclude_none=True),
            }
        )
        return (
            coerce_token_limit(model_info.get("max_input_tokens")),
            coerce_token_limit(model_info.get("max_output_tokens")),
        )

    def _core_get_configured_mode(self, model_name: str) -> "str | None":
        """Return the mode explicitly configured for a concrete deployment."""
        deployment: Final = self._core_get_deployment_by_model_group_name(model_group_name=model_name)
        if deployment is None:
            return None

        mode: Final = deployment.model_info.get("mode")
        if isinstance(mode, str) and mode.strip():
            return mode
        return None

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

        display_name: Final = deployment.model_info.get("display_name")
        if isinstance(display_name, str) and display_name.strip():
            return display_name
        return None

    def _core_get_configured_service_tiers(self, model_name: str, team_id: str | None = None) -> tuple[object, ...]:
        """
        Return the service_tiers value each deployment a request from team_id
        to model_name can route to (see _routable_deployments) configures in
        its model_info, unvalidated and in model_list order, None for a
        deployment that sets none; the caller validates the shape it expects
        and decides how the deployments combine.

        Returns an empty tuple for wildcard-expanded or unknown names.
        """
        return tuple(
            _configured_model_info(deployment).get("service_tiers")
            for deployment in self._routable_deployments(model_name, team_id)
        )

    async def _core_get_model_group_usage(self, model_group: str) -> tuple[int | None, int | None]:
        """
        Returns current tpm/rpm usage for model group

        Parameters:
        - model_group: str - the received model name from the user (can be a wildcard route).

        Returns:
        - usage: Tuple[tpm, rpm]
        """
        dt: Final = get_utc_datetime()
        current_minute: Final = dt.strftime("%H-%M")  # use the same timezone regardless of system clock
        tpm_keys: Final[list[str]] = []  # mutable-ok: ported from the Python core unchanged
        rpm_keys: Final[list[str]] = []  # mutable-ok: ported from the Python core unchanged

        model_list: Final = self._core_get_model_list(model_name=model_group)
        if model_list is None:  # no matching deployments
            return None, None

        for model in model_list:
            id: str | None = model.get("model_info", {}).get("id")
            litellm_model: str | None = model["litellm_params"].get(
                "model"
            )  # USE THE MODEL SENT TO litellm.completion() - consistent with how global_router cache is written.
            if id is None or litellm_model is None:
                continue
            tpm_keys.append(
                RouterCacheEnum.TPM.value.format(
                    id=id,
                    model=litellm_model,
                    current_minute=current_minute,
                )
            )
            rpm_keys.append(
                RouterCacheEnum.RPM.value.format(
                    id=id,
                    model=litellm_model,
                    current_minute=current_minute,
                )
            )
        combined_tpm_rpm_keys: Final = tpm_keys + rpm_keys

        with service_target(ROUTER_USAGE_TARGET):
            combined_tpm_rpm_values: Final = await self._core_cache.async_batch_get_cache(keys=combined_tpm_rpm_keys)
        if combined_tpm_rpm_values is None:
            return None, None

        rpm_start: Final = len(tpm_keys)
        tpm_usage_list: Final[list | None] = combined_tpm_rpm_values[:rpm_start]  # mutable-ok: ported from the Python core unchanged  # fmt: skip
        rpm_usage_list: Final[list | None] = combined_tpm_rpm_values[rpm_start:]  # mutable-ok: ported from the Python core unchanged  # fmt: skip

        ## TPM
        tpm_usage: int | None = None  # rebind-ok: ported from the Python core unchanged
        if tpm_usage_list is not None:
            for t in tpm_usage_list:
                if isinstance(t, int):
                    if tpm_usage is None:
                        tpm_usage = 0
                    tpm_usage += t
        ## RPM
        rpm_usage: int | None = None  # rebind-ok: ported from the Python core unchanged
        if rpm_usage_list is not None:
            for t in rpm_usage_list:
                if isinstance(t, int):
                    if rpm_usage is None:
                        rpm_usage = 0
                    rpm_usage += t
        return tpm_usage, rpm_usage

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
            itpm_keys.append(
                RouterCacheEnum.ITPM.value.format(
                    id=model_id,
                    model=litellm_model,
                    current_minute=current_minute,
                )
            )
            otpm_keys.append(
                RouterCacheEnum.OTPM.value.format(
                    id=model_id,
                    model=litellm_model,
                    current_minute=current_minute,
                )
            )

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
        model_group_info: Final = self._cached_get_model_group_info(model_group)

        returned_dict: Final[dict[str, int]] = {}  # mutable-ok: ported from the Python core unchanged

        # ITPM/OTPM groups emit input/output token headers, but they may also set
        # tpm/rpm, so build both sets rather than returning early - clients and
        # prometheus gauges that read the standard headers still get data.
        if model_group_info is not None and (model_group_info.itpm is not None or model_group_info.otpm is not None):
            current_itpm, current_otpm = await self.get_model_group_io_token_usage(model_group)
            returned_dict.update(
                build_io_token_rate_limit_headers(
                    itpm_limit=model_group_info.itpm,
                    otpm_limit=model_group_info.otpm,
                    current_itpm=current_itpm,
                    current_otpm=current_otpm,
                )
            )

        tpm_limit: Final = model_group_info.tpm if model_group_info is not None else None
        rpm_limit: Final = model_group_info.rpm if model_group_info is not None else None

        if tpm_limit is None and rpm_limit is None:
            return returned_dict

        current_tpm, current_rpm = await self._core_get_model_group_usage(model_group)

        if tpm_limit is not None:
            returned_dict["x-ratelimit-remaining-tokens"] = tpm_limit - (current_tpm or 0)
            returned_dict["x-ratelimit-limit-tokens"] = tpm_limit
        if rpm_limit is not None:
            returned_dict["x-ratelimit-remaining-requests"] = rpm_limit - (current_rpm or 0)
            returned_dict["x-ratelimit-limit-requests"] = rpm_limit

        return returned_dict

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
