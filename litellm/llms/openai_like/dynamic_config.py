"""
Dynamic configuration class generator for JSON-based providers.
"""

import json
from abc import ABC, abstractmethod
from collections.abc import Collection, Coroutine, Mapping, Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, overload

from pydantic import TypeAdapter, ValidationError

import litellm
from litellm._logging import verbose_logger
from litellm.litellm_core_utils.get_model_cost_map import GetModelCostMap
from litellm.litellm_core_utils.prompt_templates.common_utils import (
    handle_messages_with_content_list_to_str_conversion,
    handle_messages_with_text_only_content_lists_as_str,
)
from litellm.litellm_core_utils.reasoning_effort_capability import declared_reasoning_efforts
from litellm.litellm_core_utils.reasoning_effort_strength import nearest_declared_reasoning_effort
from litellm.litellm_core_utils.url_utils import ensure_api_route, join_api_base
from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
from litellm.llms.openai_like.chat.transformation import OpenAILikeChatConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues, ResponsesAPIOptionalRequestParams

from .json_loader import AcceptedValue, SimpleProviderConfig

if TYPE_CHECKING:
    from litellm.llms.openai_like.responses.transformation import OpenAILikeResponsesConfig
    from litellm.types.llms.openai import ChatCompletionToolParam

_PAID_SEARCH_TOOLS: Final = frozenset({"web_search", "web_search_image", "image_search"})
_PAYLOAD_ADAPTER: Final[TypeAdapter[Mapping[str, object]]] = TypeAdapter(Mapping[str, object])
_TOOL_USAGE_ADAPTER: Final[TypeAdapter[Mapping[str, int]]] = TypeAdapter(Mapping[str, int])
_PARAM_NAMES_ADAPTER: Final[TypeAdapter[tuple[str, ...]]] = TypeAdapter(tuple[str, ...])


def web_search_requests_from_tool_usage(payload: object) -> int | None:
    """
    Billable search calls in a parsed OpenAI-compatible chat response whose usage carries a
    tool_usage {tool_name: count} map, as EmpirioLabs reports built-in tool invocations.
    Free tools (code_interpreter, web_extractor) are not counted.
    """
    try:
        body: Final = _PAYLOAD_ADAPTER.validate_python(payload)
        usage: Final = _PAYLOAD_ADAPTER.validate_python(body.get("usage"))
        tool_usage: Final = _TOOL_USAGE_ADAPTER.validate_python(usage.get("tool_usage"), strict=True)
    except ValidationError:
        return None
    calls: Final = sum(count for tool, count in tool_usage.items() if tool in _PAID_SEARCH_TOOLS and count > 0)
    return calls if calls > 0 else None


def _flattened_messages(
    messages: Sequence[AllMessageValues], handling: Mapping[str, bool]
) -> Sequence[AllMessageValues]:
    if handling.get("convert_content_list_to_string"):
        return handle_messages_with_content_list_to_str_conversion(list(messages))
    if handling.get("convert_text_content_list_to_string"):
        return handle_messages_with_text_only_content_lists_as_str(messages)
    return messages


def _catalog_reasoning_efforts(model: str, provider_slug: str) -> tuple[str, ...] | None:
    entry: Final = GetModelCostMap.loaded_model_cost_map().get(f"{provider_slug}/{model}")
    return declared_reasoning_efforts(entry) if entry is not None else None


def _with_declared_reasoning_effort(
    params: Mapping[str, object], *, model: str, provider_slug: str, drop_params: bool
) -> Mapping[str, object]:
    requested: Final = params.get("reasoning_effort")
    declared: Final = _catalog_reasoning_efforts(model, provider_slug)
    if declared is None or not isinstance(requested, str):
        return params
    accepted: Final = nearest_declared_reasoning_effort(requested, declared)
    if drop_params and accepted not in declared:
        return MappingProxyType({key: value for key, value in params.items() if key != "reasoning_effort"})
    return MappingProxyType({**params, "reasoning_effort": accepted})


def _values_outside_accepted(
    params: Mapping[str, object], accepted_values: Mapping[str, tuple[AcceptedValue, ...]]
) -> Mapping[str, object]:
    return MappingProxyType(
        {
            param: value
            for param, value in params.items()
            if param in accepted_values and value not in accepted_values[param]
        }
    )


def _describe(params: Mapping[str, object]) -> str:
    return ", ".join(f"{param}={json.dumps(value)}" for param, value in params.items())


def _describe_accepted(accepted_values: Mapping[str, tuple[AcceptedValue, ...]]) -> str:
    return ", ".join(
        f"{param}={' or '.join(json.dumps(value) for value in values)}" for param, values in accepted_values.items()
    )


def _as_mapping(value: object) -> Mapping[str, object] | None:
    try:
        return _PAYLOAD_ADAPTER.validate_python(value)
    except ValidationError:
        return None


def _is_json_object_format(value: object) -> bool:
    format_spec: Final = _as_mapping(value)
    return format_spec is not None and format_spec.get("type") == "json_object"


def _without_json_object_formats(params: Mapping[str, object]) -> Mapping[str, object]:
    kept: Final = {
        param: value
        for param, value in params.items()
        if not (param == "response_format" and _is_json_object_format(value))
    }
    text: Final = _as_mapping(kept.get("text"))
    if text is None or not _is_json_object_format(text.get("format")):
        return MappingProxyType(kept)
    text_without_format: Final = {key: value for key, value in text.items() if key != "format"}
    without_text: Final = {param: value for param, value in kept.items() if param != "text"}
    return MappingProxyType({**without_text, "text": text_without_format} if text_without_format else without_text)


def _params_the_provider_accepts(
    params: Mapping[str, object],
    provider: SimpleProviderConfig,
    model: str,
    drop_params: bool,
    unsupported_params: frozenset[str] = frozenset(),
    allowed_openai_params: Collection[str] = (),
) -> Mapping[str, object]:
    checked_params: Final = {param: value for param, value in params.items() if param not in allowed_openai_params}
    accepted_values: Final = provider.accepted_values_for(model)
    rejected_values: Final = _values_outside_accepted(checked_params, accepted_values)
    rejected_params: Final = {param: value for param, value in checked_params.items() if param in unsupported_params}
    if (rejected_values or rejected_params) and not drop_params:
        accepted_note: Final = (
            f"It only accepts {_describe_accepted({param: accepted_values[param] for param in rejected_values})}. "
            if rejected_values
            else ""
        )
        raise litellm.UnsupportedParamsError(
            message=f"{provider.slug} does not support {_describe({**rejected_params, **rejected_values})}. "
            f"{accepted_note}To drop unsupported openai params from the call, set `litellm.drop_params = True`",
            llm_provider=provider.slug,
            model=model,
        )
    kept: Final = MappingProxyType(
        {
            param: value
            for param, value in params.items()
            if param not in rejected_values and param not in rejected_params
        }
    )
    if drop_params and provider.special_handling_for(model).get("rejects_json_object_response_format"):
        return _without_json_object_formats(kept)
    return kept


def _model_info_reading_prompt_cache_breakpoints(model_info: object, provider_slug: str) -> Mapping[str, object]:
    from litellm.integrations.anthropic_cache_control_hook import declared_prompt_cache_breakpoint, routed_model_info

    declared: Final = routed_model_info(model_info) or MappingProxyType({})
    if declared_prompt_cache_breakpoint(declared) is not None:
        return declared
    return MappingProxyType({"litellm_provider": provider_slug, **declared, "supports_prompt_cache_breakpoint": True})


class MapsAllowedOpenAIParams(ABC):
    @abstractmethod
    def map_openai_params_allowing(
        self,
        non_default_params: dict[str, object],  # mutable-ok: same caller dicts map_openai_params takes
        optional_params: dict[str, object],  # mutable-ok: filled in place like map_openai_params
        model: str,
        drop_params: bool,
        allowed_openai_params: Collection[str],
    ) -> dict[str, object]: ...  # mutable-ok: get_optional_params keeps building on the mapped params


class MapsAllowedResponsesParams(ABC):
    @abstractmethod
    def map_openai_params_for_deployment(
        self,
        response_api_optional_params: ResponsesAPIOptionalRequestParams,
        model: str,
        drop_params: bool,
        *,
        declared_reasoning_efforts: Collection[str] | None,
        allowed_openai_params: Collection[str] = (),
    ) -> Mapping[str, object]: ...

    def adjust_mapped_params(self, params: Mapping[str, object], model: str, drop_params: bool) -> Mapping[str, object]:
        return params


def create_config_class(provider: SimpleProviderConfig):
    """Generate config class dynamically from JSON configuration"""

    # Choose base class
    base_class: Final[type] = OpenAIGPTConfig if provider.base_class == "openai_gpt" else OpenAILikeChatConfig

    class JSONProviderConfig(base_class, MapsAllowedOpenAIParams):
        def get_attribution_headers(self) -> Mapping[str, str]:
            return provider.get_attribution_headers()

        def _flattens_content_lists(self) -> bool:
            return bool(provider.special_handling.get("convert_content_list_to_string"))

        @overload
        def _transform_messages(
            self, messages: list[AllMessageValues], model: str, is_async: Literal[True]
        ) -> Coroutine[object, object, list[AllMessageValues]]: ...

        @overload
        def _transform_messages(
            self,
            messages: list[AllMessageValues],
            model: str,
            is_async: Literal[False] = False,
        ) -> list[AllMessageValues]: ...

        def _transform_messages(
            self, messages: list[AllMessageValues], model: str, is_async: bool = False
        ) -> list[AllMessageValues] | Coroutine[object, object, list[AllMessageValues]]:
            """Transform messages based on special_handling config"""
            flattened: Final = _flattened_messages(messages, provider.special_handling_for(model))
            if is_async:
                return super()._transform_messages(messages=list(flattened), model=model, is_async=True)
            else:
                return super()._transform_messages(messages=list(flattened), model=model, is_async=False)

        def _should_preserve_cache_control_for_endpoint(
            self,
            custom_llm_provider: str | None,
            api_base: str | None,
        ) -> bool:
            return provider.special_handling.get(
                "preserve_cache_control", False
            ) or super()._should_preserve_cache_control_for_endpoint(custom_llm_provider, api_base)

        def _messages_and_tools_in_cache_mark_dialect(
            self,
            model: str,
            messages: Sequence[AllMessageValues],
            tools: Sequence["ChatCompletionToolParam"] | None,
            prompt_cache_options: object,
            litellm_params: Mapping[str, object],
            request_params: Mapping[str, object],
        ) -> tuple[Sequence[AllMessageValues], Sequence["ChatCompletionToolParam"] | None]:
            return super()._messages_and_tools_in_cache_mark_dialect(
                model,
                messages,
                tools,
                prompt_cache_options,
                MappingProxyType(
                    {
                        **litellm_params,
                        "model_info": _model_info_reading_prompt_cache_breakpoints(
                            litellm_params.get("model_info"), provider.slug
                        ),
                    }
                )
                if provider.special_handling_for(model).get("reads_prompt_cache_breakpoint")
                else litellm_params,
                request_params,
            )

        def _get_openai_compatible_provider_info(
            self, api_base: str | None, api_key: str | None
        ) -> tuple[str | None, str | None]:
            return self.get_model_provider_info(None, api_base, api_key)

        def get_openai_compatible_provider_info(
            self,
            api_base: str | None,
            api_key: str | None,
        ) -> tuple[str | None, str | None]:
            return self._get_openai_compatible_provider_info(api_base, api_key)

        def get_model_provider_info(
            self, model: str | None, api_base: str | None, api_key: str | None
        ) -> tuple[str | None, str | None]:
            resolved_base: Final = (
                api_base
                or (get_secret_str(provider.api_base_env) if provider.api_base_env else None)
                or provider.default_base_url(model)
            )
            resolved_key: Final = api_key or provider.api_key_from(get_secret_str)
            return resolved_base, resolved_key

        def get_complete_url(
            self,
            api_base: str | None,
            api_key: str | None,
            model: str,
            optional_params: Mapping[str, object],
            litellm_params: Mapping[str, object],
            stream: bool | None = None,
        ) -> str:
            """Build complete URL for the API endpoint"""
            return ensure_api_route(api_base or provider.default_base_url(model), "chat/completions")

        def get_supported_openai_params(self, model: str) -> list[str]:
            """Get supported OpenAI params, excluding tool-related params for models
            that don't support function calling."""
            from litellm.exceptions import ModelNotMappedError
            from litellm.utils import get_model_info_helper, supports_function_calling, supports_reasoning

            inherited_params: Final = _PARAM_NAMES_ADAPTER.validate_python(
                super().get_supported_openai_params(model=model)
            )
            supported_params: Final = [
                *(param for param in inherited_params if param not in provider.unsupported_params),
                *(param for param in provider.extra_supported_params if param not in inherited_params),
            ]

            # Check if supports_function_calling was mocked (e.g. in regression tests)
            is_mocked: Final = hasattr(supports_function_calling, "mock_calls")

            def _check_supports_fc() -> bool:
                if is_mocked:
                    return bool(supports_function_calling(model=model, custom_llm_provider=provider.slug))

                try:
                    model_info: Final = get_model_info_helper(model=model, custom_llm_provider=provider.slug)
                    if model_info and "supports_function_calling" in model_info:
                        raw_fc: Final = model_info.get("supports_function_calling")
                        if raw_fc is not None:
                            # State 3: Model row in catalog explicitly declares capability -> takes precedence
                            return bool(raw_fc)
                except (ModelNotMappedError, KeyError, ValueError):
                    pass

                # State 4 (provider explicitly disabled) vs State 1 & 2 (unmapped / missing metadata):
                # Preserves tool parameters unless provider_info explicitly sets supports_function_calling to False
                return not (provider.provider_info and provider.provider_info.get("supports_function_calling") is False)

            _supports_fc: Final = _check_supports_fc()

            if not _supports_fc:
                tool_params: Final = [
                    "tools",
                    "tool_choice",
                    "function_call",
                    "functions",
                    "parallel_tool_calls",
                ]
                for param in tool_params:
                    if param in supported_params:
                        supported_params.remove(param)
                verbose_logger.debug(
                    "Model %s on provider %s does not support function calling — removed tool-related params from supported params.",
                    model,
                    provider.slug,
                )

            _supports_reasoning: Final = supports_reasoning(model=model, custom_llm_provider=provider.slug)
            if _supports_reasoning and "reasoning_effort" not in supported_params:
                supported_params.append("reasoning_effort")

            return supported_params

        def map_openai_params_allowing(
            self,
            non_default_params: dict[str, object],  # mutable-ok: same caller dicts map_openai_params takes
            optional_params: dict[str, object],  # mutable-ok: filled in place like map_openai_params
            model: str,
            drop_params: bool,
            allowed_openai_params: Collection[str],
        ) -> dict[str, object]:  # mutable-ok: get_optional_params keeps building on the mapped params
            return self.map_openai_params(
                non_default_params, optional_params, model, drop_params, allowed_openai_params=allowed_openai_params
            )

        def map_openai_params(
            self,
            non_default_params: Mapping[str, object],
            optional_params: dict,
            model: str,
            drop_params: bool,
            allowed_openai_params: Collection[str] = (),
        ) -> dict:
            """Apply parameter mappings and constraints"""

            supported_params: Final = self.get_supported_openai_params(model)
            caller_params: Final = _PAYLOAD_ADAPTER.validate_python(non_default_params)
            requested_params: Final = (
                _with_declared_reasoning_effort(
                    caller_params,
                    model=model,
                    provider_slug=provider.slug,
                    drop_params=drop_params or litellm.drop_params is True,
                )
                if provider.special_handling_for(model).get("round_reasoning_effort_to_declared_levels")
                else caller_params
            )
            accepted_params: Final = _params_the_provider_accepts(
                requested_params,
                provider,
                model,
                drop_params or litellm.drop_params is True,
                allowed_openai_params=allowed_openai_params,
            )

            # Apply supported params
            for param, value in accepted_params.items():
                # Check parameter mappings first
                if param in provider.param_mappings:
                    optional_params[provider.param_mappings[param]] = value
                elif param in supported_params:
                    optional_params[param] = value

            # Apply temperature constraints if present
            if "temperature" in optional_params:
                temp = optional_params["temperature"]
                constraints: Final = provider.constraints

                # Clamp to max
                if "temperature_max" in constraints:
                    temp = min(temp, constraints["temperature_max"])

                # Clamp to min
                if "temperature_min" in constraints:
                    temp = max(temp, constraints["temperature_min"])

                # Special case: temperature_min_with_n_gt_1
                if "temperature_min_with_n_gt_1" in constraints:
                    n: Final = optional_params.get("n", 1)
                    if n > 1 and temp < constraints["temperature_min_with_n_gt_1"]:
                        temp = constraints["temperature_min_with_n_gt_1"]

                optional_params["temperature"] = temp

            return optional_params

        def transform_parsed_response_dict(
            self,
            parsed_response: dict[str, object],  # mutable-ok: BaseConfig signature
        ) -> dict[str, object]:  # mutable-ok: BaseConfig signature
            """
            Carry the provider's usage.tool_usage search count as usage.server_tool_use.web_search_requests,
            the field the built-in tool cost path bills per call.
            """
            repaired: Final = _PAYLOAD_ADAPTER.validate_python(
                super().transform_parsed_response_dict(parsed_response)  # pyright: ignore[reportUnknownMemberType]  # base_class is picked at runtime
            )
            if not provider.special_handling.get("web_search_requests_from_tool_usage"):
                return {**repaired}
            web_search_requests: Final = web_search_requests_from_tool_usage(repaired)
            if web_search_requests is None:
                return {**repaired}
            usage: Final = _PAYLOAD_ADAPTER.validate_python(repaired["usage"])
            return {**repaired, "usage": {**usage, "server_tool_use": {"web_search_requests": web_search_requests}}}

        @property
        def custom_llm_provider(self) -> str | None:
            return provider.slug

    return JSONProviderConfig


_responses_config_cache: Final[dict[str, "type[OpenAILikeResponsesConfig]"]] = {}


def create_responses_config_class(provider: SimpleProviderConfig) -> "type[OpenAILikeResponsesConfig]":
    """Generate a Responses API config class dynamically from JSON configuration.

    Parallel to create_config_class() but for /v1/responses endpoints.
    Classes are cached per provider slug to avoid regeneration on every request.
    """
    if provider.slug in _responses_config_cache:
        return _responses_config_cache[provider.slug]

    from litellm.integrations.anthropic_cache_control_hook import deployment_model_info
    from litellm.llms.openai_like.responses.transformation import (
        OpenAILikeResponsesConfig,
    )
    from litellm.types.llms.openai import ResponseInputParam
    from litellm.types.router import GenericLiteLLMParams

    class JSONProviderResponsesConfig(OpenAILikeResponsesConfig, MapsAllowedResponsesParams):
        @property
        def custom_llm_provider(self):
            return provider.slug

        def validate_environment(
            self,
            headers: dict,
            model: str,
            litellm_params: GenericLiteLLMParams | None,
        ) -> dict:
            litellm_params = litellm_params or GenericLiteLLMParams()
            api_key: Final = litellm_params.api_key or provider.api_key_from(get_secret_str)
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            return headers

        def get_complete_url(
            self,
            api_base: str | None,
            litellm_params: Mapping[str, object],
        ) -> str:
            resolved_base: Final = (
                api_base
                or (get_secret_str(provider.api_base_env) if provider.api_base_env else None)
                or provider.base_url
            )
            return join_api_base(resolved_base, "responses")

        def _reads_prompt_cache_breakpoints(
            self, model: str, litellm_params: GenericLiteLLMParams, prompt_cache_options: object
        ) -> bool:
            return super()._reads_prompt_cache_breakpoints(
                model,
                litellm_params.model_copy(
                    update={
                        "model_info": dict(
                            _model_info_reading_prompt_cache_breakpoints(
                                deployment_model_info(litellm_params), provider.slug
                            )
                        )
                    }
                )
                if provider.special_handling_for(model).get("reads_prompt_cache_breakpoint")
                else litellm_params,
                prompt_cache_options,
            )

        def map_openai_params_for_deployment(
            self,
            response_api_optional_params: ResponsesAPIOptionalRequestParams,
            model: str,
            drop_params: bool,
            *,
            declared_reasoning_efforts: Collection[str] | None,
            allowed_openai_params: Collection[str] = (),
        ) -> Mapping[str, object]:
            return self.adjust_mapped_params(
                {
                    **_params_the_provider_accepts(
                        super().map_openai_params_for_deployment(
                            response_api_optional_params,
                            model,
                            drop_params,
                            declared_reasoning_efforts=declared_reasoning_efforts,
                        ),
                        provider,
                        model,
                        drop_params or litellm.drop_params is True,
                        provider.responses_unsupported_params,
                        allowed_openai_params,
                    )
                },
                model,
                drop_params,
            )

        def transform_responses_api_request(
            self,
            model: str,
            input: str | ResponseInputParam,
            response_api_optional_request_params: dict,
            litellm_params: GenericLiteLLMParams,
            headers: dict,
        ) -> dict:
            if provider.special_handling.get("force_store_false"):
                response_api_optional_request_params["store"] = False
            return super().transform_responses_api_request(
                model=model,
                input=input,
                response_api_optional_request_params=response_api_optional_request_params,
                litellm_params=litellm_params,
                headers=headers,
            )

    _responses_config_cache[provider.slug] = JSONProviderResponsesConfig
    return JSONProviderResponsesConfig
