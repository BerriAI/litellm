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
from litellm.litellm_core_utils.prompt_templates.common_utils import (
    handle_messages_with_content_list_to_str_conversion,
)
from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
from litellm.llms.openai_like.chat.transformation import OpenAILikeChatConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues, ResponsesAPIOptionalRequestParams

from .json_loader import AcceptedValue, SimpleProviderConfig

if TYPE_CHECKING:
    from litellm.llms.openai_like.responses.transformation import OpenAILikeResponsesConfig
    from litellm.types.llms.openai import ChatCompletionToolParam

_PARAM_NAMES_ADAPTER: Final[TypeAdapter[tuple[str, ...]]] = TypeAdapter(tuple[str, ...])
_PAYLOAD_ADAPTER: Final[TypeAdapter[Mapping[str, object]]] = TypeAdapter(Mapping[str, object])


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
    def map_openai_params_allowing(
        self,
        response_api_optional_params: ResponsesAPIOptionalRequestParams,
        model: str,
        drop_params: bool,
        allowed_openai_params: Collection[str],
    ) -> Mapping[str, object]: ...

    def adjust_mapped_params(self, params: Mapping[str, object], model: str, drop_params: bool) -> Mapping[str, object]:
        return params


def create_config_class(provider: SimpleProviderConfig):
    """Generate config class dynamically from JSON configuration"""

    # Choose base class
    base_class: Final[type] = OpenAIGPTConfig if provider.base_class == "openai_gpt" else OpenAILikeChatConfig

    class JSONProviderConfig(base_class, MapsAllowedOpenAIParams):
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

            # Handle content list to string conversion if configured
            if self._flattens_content_lists():
                messages = handle_messages_with_content_list_to_str_conversion(messages)

            if is_async:
                return super()._transform_messages(messages=messages, model=model, is_async=True)
            else:
                return super()._transform_messages(messages=messages, model=model, is_async=False)

        def _messages_and_tools_in_cache_mark_dialect(
            self,
            model: str,
            messages: Sequence[AllMessageValues],
            tools: Sequence["ChatCompletionToolParam"] | None,
            prompt_cache_options: object,
            litellm_params: Mapping[str, object],
        ) -> tuple[Sequence[AllMessageValues], Sequence["ChatCompletionToolParam"] | None]:
            if not provider.special_handling_for(model).get("reads_prompt_cache_breakpoint"):
                return super()._messages_and_tools_in_cache_mark_dialect(
                    model, messages, tools, prompt_cache_options, litellm_params
                )
            from litellm.integrations.anthropic_cache_control_hook import AnthropicCacheControlHook

            return self.remove_cache_control_flag_from_messages_and_tools(
                model=model,
                messages=[
                    AnthropicCacheControlHook.cache_control_as_prompt_cache_breakpoints(message) for message in messages
                ],
                tools=None if tools is None else list(tools),
            )

        def _get_openai_compatible_provider_info(
            self, api_base: str | None, api_key: str | None
        ) -> tuple[str | None, str | None]:
            """Get API base and key from JSON config"""

            # Resolve base URL
            resolved_base = api_base
            if not resolved_base and provider.api_base_env:
                resolved_base = get_secret_str(provider.api_base_env)
            if not resolved_base:
                resolved_base = provider.base_url

            # Resolve API key
            resolved_key: Final = api_key or provider.api_key_from(get_secret_str)

            return resolved_base, resolved_key

        def get_openai_compatible_provider_info(
            self,
            api_base: str | None,
            api_key: str | None,
        ) -> tuple[str | None, str | None]:
            return self._get_openai_compatible_provider_info(api_base, api_key)

        def get_complete_url(
            self,
            api_base: str | None,
            api_key: str | None,
            model: str,
            optional_params: dict,
            litellm_params: dict,
            stream: bool | None = None,
        ) -> str:
            """Build complete URL for the API endpoint"""
            if not api_base:
                api_base = provider.base_url

            if api_base is None:
                raise ValueError(f"api_base is required for provider {provider.slug}")

            if not api_base.endswith("/chat/completions"):
                api_base = f"{api_base}/chat/completions"

            return api_base

        def get_supported_openai_params(self, model: str) -> list:
            """Get supported OpenAI params, excluding tool-related params for models
            that don't support function calling."""
            from litellm.utils import supports_function_calling, supports_reasoning

            inherited_params: Final = _PARAM_NAMES_ADAPTER.validate_python(
                super().get_supported_openai_params(model=model)
            )
            supported_params: Final = [
                *(param for param in inherited_params if param not in provider.unsupported_params),
                *(param for param in provider.extra_supported_params if param not in inherited_params),
            ]

            _supports_fc: Final = supports_function_calling(model=model, custom_llm_provider=provider.slug)

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
            non_default_params: dict,
            optional_params: dict,
            model: str,
            drop_params: bool,
            allowed_openai_params: Collection[str] = (),
        ) -> dict:
            """Apply parameter mappings and constraints"""

            supported_params: Final = self.get_supported_openai_params(model)
            accepted_params: Final = _params_the_provider_accepts(
                non_default_params,
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

        @property
        def custom_llm_provider(self) -> str | None:
            return provider.slug

    return JSONProviderConfig


_responses_config_cache: Final[dict] = {}


def create_responses_config_class(provider: SimpleProviderConfig) -> "type[OpenAILikeResponsesConfig]":
    """Generate a Responses API config class dynamically from JSON configuration.

    Parallel to create_config_class() but for /v1/responses endpoints.
    Classes are cached per provider slug to avoid regeneration on every request.
    """
    if provider.slug in _responses_config_cache:
        return _responses_config_cache[provider.slug]

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
            litellm_params: dict,
        ) -> str:
            if not api_base:
                api_base = (
                    get_secret_str(provider.api_base_env) if provider.api_base_env else None
                ) or provider.base_url

            if api_base is None:
                raise ValueError(f"api_base is required for provider {provider.slug}")

            api_base = api_base.rstrip("/")
            return f"{api_base}/responses"

        def starts_background_runs(self, model: str, api_base: str | None) -> bool:
            return provider.special_handling_for(model).get("starts_background_runs") is True

        def _reads_prompt_cache_breakpoints(
            self, model: str, litellm_params: GenericLiteLLMParams, prompt_cache_options: object
        ) -> bool:
            return bool(
                provider.special_handling_for(model).get("reads_prompt_cache_breakpoint")
            ) or super()._reads_prompt_cache_breakpoints(model, litellm_params, prompt_cache_options)

        def map_openai_params(
            self,
            response_api_optional_params: ResponsesAPIOptionalRequestParams,
            model: str,
            drop_params: bool,
        ) -> dict[str, object]:  # mutable-ok: return type fixed by the override contract
            return {**self.map_openai_params_allowing(response_api_optional_params, model, drop_params, ())}

        def map_openai_params_allowing(
            self,
            response_api_optional_params: ResponsesAPIOptionalRequestParams,
            model: str,
            drop_params: bool,
            allowed_openai_params: Collection[str],
        ) -> Mapping[str, object]:
            return self.adjust_mapped_params(
                {
                    **_params_the_provider_accepts(
                        super().map_openai_params(response_api_optional_params, model, drop_params),
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
